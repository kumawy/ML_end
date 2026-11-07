# %% [markdown]
# # Used-car asking prices in Kazakhstan: endterm
# Run this file from top to bottom. All results use the original midterm CSV.
# Supervised prediction, clustering/PCA, and ANN are separate experiments.
from __future__ import annotations

import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import json
import platform
import time
from pathlib import Path
from importlib.metadata import version

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from threadpoolctl import threadpool_limits

from data import (SEED, FEATURES, NUMERIC, CATEGORICAL, CORRECTED_NUMERIC,
                  CORRECTED_CATEGORICAL, load_data, split_data, make_preprocessor)
from ann import train_ann
from unsupervised import run_unsupervised
from ensemble import Component, PredictionEnsemble
import legacy_data

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
GREEN = '#174B40'
ACCENT = '#73B99C'
LEGACY_NAMES = {'Median baseline','Linear regression','KNN midterm','Tree midterm'}
SELECTED_MODEL = 'ANN ensemble'
MODEL_ORDER = ('Median baseline', 'Linear regression', 'KNN midterm', 'Tree midterm',
               'KNN tuned', 'Tree tuned', 'Random forest tuned', 'ANN corrected', SELECTED_MODEL)


def metrics(y, prediction):
    y = np.asarray(y, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if not np.isfinite(prediction).all():
        raise ValueError('Non-finite predictions')
    errors = np.abs(y-prediction)
    relative = errors/y
    return {'mae_kzt':float(mean_absolute_error(y,prediction)),
            'rmse_kzt':float(np.sqrt(mean_squared_error(y,prediction))),
            'r2':float(r2_score(y,prediction)),
            'median_absolute_error_kzt':float(np.median(errors)),
            'median_ape':float(np.median(relative)),
            'within_10pct':float(np.mean(relative <= .10)),
            'within_20pct':float(np.mean(relative <= .20)),
            'within_30pct':float(np.mean(relative <= .30))}


def write_json(path, content):
    def convert(value):
        if isinstance(value, np.generic): return value.item()
        if isinstance(value, np.ndarray): return value.tolist()
        if isinstance(value, Path): return str(value.name)
        raise TypeError(type(value).__name__)
    # DataFrame JSON conversion safely maps genuinely missing measurements to null.
    safe = json.loads(pd.Series([content]).to_json(orient='values',default_handler=convert))[0]
    Path(path).write_text(json.dumps(safe,indent=2,ensure_ascii=False,default=convert,allow_nan=False),encoding='utf-8')


def savefig(fig,name):
    fig.tight_layout()
    fig.savefig(OUT / name,dpi=170,bbox_inches='tight')
    plt.close(fig)


def explore(train):
    """EDA is descriptive and confined to training listings."""
    bins=[0,2e6,5e6,10e6,20e6,40e6,np.inf]
    labels=['<2','2-5','5-10','10-20','20-40','40+']
    counts=pd.cut(train.price,bins=bins,labels=labels,right=False).value_counts(sort=False)
    hist=pd.DataFrame({'price_band_million_kzt':labels,'count':counts.to_numpy()})
    hist.to_csv(OUT/'eda_price_bands.csv',index=False)
    fig,ax=plt.subplots(figsize=(8,4.5));ax.bar(labels,hist['count'],color=GREEN)
    ax.set(xlabel='Asking price (million KZT)',ylabel='Training listings',title='Training price distribution')
    savefig(fig,'eda_prices.png')
    years=train.groupby('year').price.agg(['median','size']).reset_index()
    years=years.loc[(years.year>=1990)&(years['size']>=8)]
    years.to_csv(OUT/'eda_year.csv',index=False)
    fig,ax=plt.subplots(figsize=(8,4.5));ax.plot(years.year,years['median']/1e6,'o-',color=GREEN,markersize=3)
    ax.set(xlabel='Model year',ylabel='Median asking price (million KZT)',title='Prices by model year')
    savefig(fig,'eda_year.png')
    sample=train.loc[train.mileage_km.notna()].sample(min(1200,int(train.mileage_km.notna().sum())),random_state=SEED)
    sample[['mileage_km','price']].to_csv(OUT/'eda_mileage_sample.csv',index=False)
    fig,ax=plt.subplots(figsize=(8,4.5));ax.scatter(sample.mileage_km/1000,sample.price/1e6,s=12,alpha=.35,color=GREEN)
    ax.set(xlabel='Mileage (thousand km)',ylabel='Asking price (million KZT)',title='Mileage and asking price')
    savefig(fig,'eda_mileage.png')
    fuel=train.groupby('fuel').price.agg(['median','size']).reset_index()
    fuel.to_csv(OUT/'eda_fuel.csv',index=False)
    # Fourth explicit EDA plot required by the midterm regulation. Retain every
    # training observation and show small-group counts; log scale handles tails.
    ordered=fuel.sort_values('median')
    fuel_values=[train.loc[train.fuel.eq(name),'price'].to_numpy()/1e6 for name in ordered.fuel]
    fuel_labels=[f'{row.fuel}\nn={int(row["size"])}' for _,row in ordered.iterrows()]
    fig,ax=plt.subplots(figsize=(9,5))
    boxes=ax.boxplot(fuel_values,tick_labels=fuel_labels,patch_artist=True,
                     flierprops={'marker':'.','markersize':4,'alpha':.4})
    for box in boxes['boxes']:
        box.set(facecolor=ACCENT,edgecolor=GREEN)
    ax.set_yscale('log')
    ax.set_yticks([.5,1,2,5,10,20,50,100,200])
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.set(xlabel='Advertised fuel group; n is training count',
           ylabel='Asking price (million KZT, log scale)',
           title='Training asking-price distributions by fuel')
    savefig(fig,'eda_fuel.png')
    plots=[
        {'file':'eda_prices.png','interpretation':f'Training mean price {train.price.mean()/1e6:.2f}m exceeds median {train.price.median()/1e6:.2f}m KZT, indicating a high-price tail.'},
        {'file':'eda_year.png','interpretation':'Newer model years generally have higher median asking prices. Year-price association does not establish causality; displayed years require at least eight training listings.'},
        {'file':'eda_mileage.png','interpretation':'Higher observed mileage is associated with lower asking prices, with substantial spread. The chart uses a seeded sample of up to 1200 training rows with known mileage.'},
        {'file':'eda_fuel.png','interpretation':'Fuel groups have different price distributions, but also different vehicle mixes. All training rows are retained; logarithmic price scale and group counts are explicit. Electric and hybrid groups are small, so group price gaps are descriptive, not causal.'},
    ]
    return {'training_mean_price_kzt':float(train.price.mean()),'training_median_price_kzt':float(train.price.median()),
            'year_price_spearman':float(train[['year','price']].corr(method='spearman').iloc[0,1]),
            'mileage_price_spearman':float(train[['mileage_km','price']].corr(method='spearman').iloc[0,1]),
            'price_bands':hist.to_dict('records'),'fuel_groups':fuel.to_dict('records'),
            'plots':plots,'fit_scope':'Training listings only; four explicit EDA plots, each with an interpretation.'}


def supervised_experiments(X_train,y_train,X_val,y_val,legacy_train,legacy_val):
    """Compare midterm estimators and tune new candidates on training CV only."""
    def pipe(estimator,legacy=False):
        return Pipeline([('prep',legacy_data.make_preprocessor() if legacy else make_preprocessor()),('model',estimator)])
    models={
        'Median baseline':DummyRegressor(strategy='median'),
        'Linear regression':pipe(LinearRegression(),legacy=True),
        'KNN midterm':pipe(KNeighborsRegressor(n_neighbors=15,weights='distance'),legacy=True),
        'Tree midterm':pipe(DecisionTreeRegressor(max_depth=10,min_samples_leaf=3,random_state=SEED),legacy=True),
    }
    for name,model in models.items():
        print('Fit legacy baseline',name,flush=True);model.fit(legacy_train,y_train)
    cv=KFold(n_splits=3,shuffle=True,random_state=SEED)
    searches=[
        ('KNN tuned',KNeighborsRegressor(),{'model__n_neighbors':[3,5,9,15,25,35], 'model__weights':['uniform','distance'],'model__p':[1,2]}),
        ('Tree tuned',DecisionTreeRegressor(random_state=SEED),{'model__max_depth':[6,10,14,None],'model__min_samples_leaf':[3,5,10]}),
        ('Random forest tuned',RandomForestRegressor(n_estimators=200,random_state=SEED,n_jobs=2),{'model__max_features':[.7,1.0],'model__min_samples_leaf':[1,3]}),
    ]
    reports=[]
    for name,estimator,grid in searches:
        grid={**grid,'prep__cat__encode__min_frequency':[1,2,5]}
        print('Training-only 3-fold CV:',name,flush=True)
        search=GridSearchCV(pipe(estimator),grid,scoring='neg_mean_absolute_error',cv=cv,n_jobs=1,refit=True,return_train_score=True,error_score='raise')
        search.fit(X_train,y_train)
        models[name]=search.best_estimator_
        cv_frame=pd.DataFrame(search.cv_results_)
        cv_frame.to_csv(OUT/(name.lower().replace(' ','_')+'_cv.csv'),index=False)
        reports.append({'model':name,'best_parameters':search.best_params_,
                        'cv_mae_kzt':float(-search.best_score_),
                        'cv_std_mae_kzt':float(search.cv_results_['std_test_score'][search.best_index_]),
                        'candidate_count':len(cv_frame),'folds':3,'shuffle':True,'seed':SEED})
    validation={name:metrics(y_val,model.predict(legacy_val if name in LEGACY_NAMES else X_val)) for name,model in models.items()}
    return models,validation,{'cv_results':reports,'cv_protocol':'3-fold KFold, shuffle=True, seed 42, within training only',
                              'cv_score':'negative MAE in KZT','all_final_fits':'1787 training listings; validation not refitted',
                              'midterm_tree_parameters':{'max_depth':10,'min_samples_leaf':3},
                              'midterm_cv_note':'Original tree used unshuffled 3-fold CV. New searches use one shared shuffled CV.',
                              'representation':'Four original baselines use unchanged legacy_data.py. Tuned models use corrected/enriched inputs; OHE min_frequency 1/2/5 is selected inside CV.'}


def ann_experiments(X_train, y_train, X_val, y_val):
    """Train only the corrected-view ANN at the frozen OHE threshold of two."""
    prep = make_preprocessor(min_frequency=2, feature_set='corrected')
    training = prep.fit_transform(X_train)
    validation = prep.transform(X_val)
    model, report = train_ann(training, y_train.to_numpy(), validation, y_val.to_numpy(), OUT, seed=SEED)
    report.update({
        'reported_model_name': 'ANN corrected', 'input_feature_count': int(training.shape[1]),
        'selected_min_frequency': 2, 'feature_set': 'corrected', 'history_directory': '.',
        'preprocessing': {'fit_split': 'training only', 'training_rows': len(X_train),
                          'numeric_inputs': CORRECTED_NUMERIC,
                          'categorical_inputs': CORRECTED_CATEGORICAL,
                          'categorical_min_frequency': 2,
                          'numeric_transform': 'Training median imputation followed by StandardScaler',
                          'categorical_transform': 'Constant unknown imputation; dense OHE with infrequent/unknown handling'},
    })
    return model, prep, report


def build_selected_ensemble(corrected_pool, usage_pool, forest_component):
    """Reconstruct the previously selected recipe without another blend search.

    The multiplication order preserves the original flattened ensemble's
    arithmetic: five usage-MAE members, five corrected-dual members, then RF.
    """
    robust_weights = [.75 * float(weight) for weight in corrected_pool.weights] + [.25]
    return PredictionEnsemble(
        [*usage_pool.components, *corrected_pool.components, forest_component],
        [.25 * float(weight) for weight in usage_pool.weights]
        + [.75 * float(weight) for weight in robust_weights],
    )


def error_analysis(frame,split,y_train,predictions,selected):
    test=frame.loc[split['test']].copy()
    test['prediction_kzt']=predictions
    test['residual_kzt']=test.price-test.prediction_kzt
    test['absolute_error_kzt']=test.residual_kzt.abs()
    cuts=np.quantile(y_train,[.25,.5,.75])
    test['price_band']=pd.cut(test.price,[-np.inf,*cuts,np.inf],labels=['Q1','Q2','Q3','Q4'],include_lowest=True)
    price=test.groupby('price_band',observed=True).agg(count=('price','size'),mae_kzt=('absolute_error_kzt','mean'),median_error_kzt=('absolute_error_kzt','median'),mean_residual_kzt=('residual_kzt','mean')).reset_index()
    price.to_csv(OUT/'error_by_price_band.csv',index=False)
    brands=test.groupby('brand').agg(count=('price','size'),mae_kzt=('absolute_error_kzt','mean')).reset_index().sort_values('mae_kzt',ascending=False)
    brands.to_csv(OUT/'error_by_brand.csv',index=False)
    missing=test.groupby('mileage_missing').agg(count=('price','size'),mae_kzt=('absolute_error_kzt','mean')).reset_index()
    missing.to_csv(OUT/'error_by_missing_mileage.csv',index=False)
    train_counts=frame.loc[split['train'],'model'].value_counts()
    test['training_model_count']=test.model.map(train_counts).fillna(0).astype(int)
    test['model_support']=pd.cut(test.training_model_count,[-1,0,4,np.inf],labels=['unseen','rare_1_to_4','seen_5plus'])
    support=test.groupby('model_support',observed=True).agg(count=('price','size'),mae_kzt=('absolute_error_kzt','mean')).reset_index()
    support.to_csv(OUT/'error_by_model_support.csv',index=False)
    fuel=test.groupby('fuel').agg(count=('price','size'),mae_kzt=('absolute_error_kzt','mean')).reset_index()
    fuel.to_csv(OUT/'error_by_fuel.csv',index=False)
    test['age_band']=pd.cut(2025-test.year,[-1,5,10,20,np.inf],labels=['0-5','6-10','11-20','21+'])
    ages=test.groupby('age_band',observed=True).agg(count=('price','size'),mae_kzt=('absolute_error_kzt','mean')).reset_index()
    ages.to_csv(OUT/'error_by_age.csv',index=False)
    test['absolute_percentage_error']=test.absolute_error_kzt/test.price
    columns=['ad_id','brand','model','year','mileage_km','mileage_missing','fuel','training_model_count','model_support','price','prediction_kzt','residual_kzt','absolute_error_kzt','absolute_percentage_error','price_band','age_band']
    test[columns].to_csv(OUT/'selected_model_test_predictions.csv',index=False)
    test.nlargest(15,'absolute_error_kzt')[columns].to_csv(OUT/'largest_errors.csv',index=False)
    fig,ax=plt.subplots(figsize=(8,4.5));ax.bar(price.price_band.astype(str),price.mae_kzt/1e6,color=GREEN)
    ax.set(xlabel='True-price quartile (cutoffs from training)',ylabel='Test MAE (million KZT)',title=f'Errors by price segment: {selected}')
    savefig(fig,'error_by_price.png')
    fig,ax=plt.subplots(figsize=(6,5));ax.scatter(test.price/1e6,test.prediction_kzt/1e6,s=16,alpha=.4,color=GREEN)
    limit=max(test.price.max(),test.prediction_kzt.max())/1e6
    ax.plot([0,limit],[0,limit],color='#CE8767')
    ax.set(xlabel='Actual asking price (million KZT)',ylabel='Predicted asking price (million KZT)',title=selected)
    savefig(fig,'test_predictions.png')
    return {'selected_model':selected,'quartile_cutoffs_kzt':cuts.tolist(),'price_bands':price.to_dict('records'),
            'missing_mileage':missing.to_dict('records'),'age_bands':ages.to_dict('records'),
            'model_support':support.to_dict('records'),'fuel_groups':fuel.to_dict('records'),
            'largest_errors':test.nlargest(5,'absolute_error_kzt')[columns].to_dict('records'),
            'mean_residual_kzt':float(test.residual_kzt.mean()),
            'residual_sign':'actual minus prediction; positive means underestimation'}


def main():
    from robust_ann import fit_robust_pool
    from structured_features import add_structured_features
    from dataset_features import add_dataset_features
    from dataset_export import export_dataset
    start = time.perf_counter()
    OUT.mkdir(exist_ok=True)
    np.random.seed(SEED)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False})
    frame, audit = load_data(ROOT)
    split = split_data(frame)
    dataset_report = export_dataset(frame, split, ROOT / 'dataset', ROOT / 'kolesa_cars_2025.csv')
    legacy_frame, _ = legacy_data.load_data(ROOT)
    assert frame.index.equals(legacy_frame.index)
    assert frame.ad_id.equals(legacy_frame.ad_id)
    assert frame.price.equals(legacy_frame.price)
    print('Source data:', len(frame), 'unique used listings; original IDs/targets retained', flush=True)
    split_manifest = [
        {'ad_id': str(frame.loc[i, 'ad_id']), 'source_row': int(i), 'split': name}
        for name, indices in split.items() for i in indices
    ]
    pd.DataFrame(split_manifest).to_csv(OUT / 'split_manifest.csv', index=False)
    X_train = frame.loc[split['train'], FEATURES]
    y_train = frame.loc[split['train'], 'price']
    X_val = frame.loc[split['validation'], FEATURES]
    y_val = frame.loc[split['validation'], 'price']
    legacy_train = legacy_frame.loc[split['train'], legacy_data.FEATURES]
    legacy_val = legacy_frame.loc[split['validation'], legacy_data.FEATURES]
    train = frame.loc[split['train']]
    eda = explore(train)
    # Displacement geometry is meaningful for combustion cars. Electric rows
    # remain in supervised prediction, but are excluded from this numeric view.
    cluster_train = train.loc[~train.fuel.eq('electric')]
    print('Clustering/PCA:', len(cluster_train), 'non-electric training listings', flush=True)
    unsupervised = run_unsupervised(cluster_train, OUT, seed=SEED)
    unsupervised['analysis_subset'] = 'Non-electric training listings only; engine displacement is not applicable to explicit electric listings.'
    unsupervised['excluded_electric_rows'] = int(len(train) - len(cluster_train))
    write_json(OUT / 'unsupervised_report.json', unsupervised)

    models, validation, supervised = supervised_experiments(
        X_train, y_train, X_val, y_val, legacy_train, legacy_val)
    corrected_ann, corrected_prep, ann_report = ann_experiments(X_train, y_train, X_val, y_val)
    validation_predictions = {
        name: model.predict(legacy_val if name in LEGACY_NAMES else X_val)
        for name, model in models.items()
    }
    validation_predictions['ANN corrected'] = corrected_ann.predict(corrected_prep.transform(X_val))
    validation['ANN corrected'] = metrics(y_val, validation_predictions['ANN corrected'])
    forest = Component(models['Random forest tuned'], 'enhanced')
    # These two references are required by the original checkpoint scores;
    # neither is a new comparison candidate or a newly searched mixture.
    reference = PredictionEnsemble([Component(corrected_ann, 'enhanced', corrected_prep), forest], [.75, .25])
    reference_prediction = reference.predict({'enhanced': X_val})
    print('Fixed five-seed corrected/steering ANN pool', flush=True)
    corrected_pools, robust_report = fit_robust_pool(
        frame.loc[split['train']], y_train.to_numpy(),
        frame.loc[split['validation']], y_val.to_numpy(),
        reference_prediction, OUT / 'robust_ann', seed=SEED)
    corrected_pool = corrected_pools['ANN pool dual']
    robust_reference = PredictionEnsemble(
        [*corrected_pool.components, forest],
        [.75 * float(weight) for weight in corrected_pool.weights] + [.25])
    structured_val = add_structured_features(frame.loc[split['validation']])
    robust_prediction = robust_reference.predict({'enhanced': structured_val})
    print('Fixed five-seed age/usage ANN pool', flush=True)
    enriched_frame = add_dataset_features(frame)
    usage_pools, usage_report = fit_robust_pool(
        enriched_frame.loc[split['train']], y_train.to_numpy(),
        enriched_frame.loc[split['validation']], y_val.to_numpy(),
        robust_prediction, OUT / 'usage_ann', seed=SEED,
        additional_numeric=('log_vehicle_age', 'log_annual_mileage_proxy'))
    selected = build_selected_ensemble(corrected_pool, usage_pools['ANN pool MAE'], forest)
    enriched_val = add_structured_features(enriched_frame.loc[split['validation']])
    validation_predictions[SELECTED_MODEL] = selected.predict({'enhanced': enriched_val})
    validation[SELECTED_MODEL] = metrics(y_val, validation_predictions[SELECTED_MODEL])
    robust_report['artifact_base_directory'] = 'robust_ann'
    usage_report['artifact_base_directory'] = 'usage_ann'
    ensemble_report = {
        'model': SELECTED_MODEL, 'recipe_status': 'Frozen before this cleanup run; no new blend search',
        'formula': '0.25 * usage_MAE_pool + 0.75 * (0.75 * corrected_dual_pool + 0.25 * RF)',
        'reference_for_corrected_checkpoints': '0.75 * ANN corrected + 0.25 * Random forest tuned',
        'reference_for_usage_checkpoints': '0.75 * corrected_dual_pool + 0.25 * Random forest tuned',
        'pool_seeds': [11, 23, 42, 71, 101],
        'component_order': ['usage_MAE_seed_' + str(seed) for seed in [11, 23, 42, 71, 101]]
                           + ['corrected_dual_seed_' + str(seed) for seed in [11, 23, 42, 71, 101]]
                           + ['Random forest tuned'],
        'component_weights': selected.weights,
        'checkpoint_selection': {'corrected_pool': 'dual_score', 'usage_pool': 'mae_kzt'},
        'test_data_used_for_this_run_selection': False,
    }
    write_json(OUT / 'ensemble_report.json', ensemble_report)
    validation_predictions = {name: validation_predictions[name] for name in MODEL_ORDER}
    pd.DataFrame({'ad_id': frame.loc[split['validation'], 'ad_id'].to_numpy(),
                  'actual_price_kzt': y_val.to_numpy(), **validation_predictions}).to_csv(
                      OUT / 'all_validation_predictions.csv', index=False)
    policy = ('Historical test errors motivated earlier feature-engineering hypotheses. '
              'This cleanup preserves the previously selected recipe. Training CV and validation control '
              'fitting/checkpoints; there is no new blend search or test-based acceptance. '
              'This reused test is retrospective and cannot establish independent generalization.')
    selection = {
        'selected_model': SELECTED_MODEL, 'selection_metric': 'Previously selected frozen recipe; no new model or blend search',
        'seed': SEED, 'validation': validation, 'test_policy': policy,
        'frozen_recipe': ensemble_report,
    }
    # Write the recipe and all validation evidence before accessing test inputs.
    write_json(OUT / 'selection_before_test.json', selection)
    print('Frozen selected recipe:', SELECTED_MODEL, '; evaluating retrospective benchmark once', flush=True)
    X_test = frame.loc[split['test'], FEATURES]
    y_test = frame.loc[split['test'], 'price']
    legacy_test = legacy_frame.loc[split['test'], legacy_data.FEATURES]
    test_predictions = {
        name: model.predict(legacy_test if name in LEGACY_NAMES else X_test)
        for name, model in models.items()
    }
    test_predictions['ANN corrected'] = corrected_ann.predict(corrected_prep.transform(X_test))
    enriched_test = add_structured_features(enriched_frame.loc[split['test']])
    test_predictions[SELECTED_MODEL] = selected.predict({'enhanced': enriched_test})
    test_predictions = {name: test_predictions[name] for name in MODEL_ORDER}
    rows = [
        {'model': name, **{'validation_' + key: value for key, value in validation[name].items()},
         **{'test_' + key: value for key, value in metrics(y_test, prediction).items()},
         'selected': name == SELECTED_MODEL,
         'representation': ('mixed ANN/RF' if name == SELECTED_MODEL else
                            ('legacy' if name in LEGACY_NAMES else
                             ('corrected' if name == 'ANN corrected' else 'enhanced')))}
        for name, prediction in test_predictions.items()
    ]
    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUT / 'model_comparison.csv', index=False)
    pd.DataFrame({'ad_id': frame.loc[split['test'], 'ad_id'].to_numpy(),
                  'actual_price_kzt': y_test.to_numpy(), **test_predictions}).to_csv(
                      OUT / 'all_test_predictions.csv', index=False)
    errors = error_analysis(frame, split, y_train, test_predictions[SELECTED_MODEL], SELECTED_MODEL)
    best = next(row for row in rows if row['selected'])
    baseline = next(row for row in rows if row['model'] == 'Median baseline')
    midterm = next(row for row in rows if row['model'] == 'KNN midterm')
    improvement = 1 - best['test_mae_kzt'] / baseline['test_mae_kzt']
    environment = {
        'python': platform.python_version(), 'platform': platform.platform(), 'seed': SEED,
        'packages': {name: version(name) for name in
                     ['numpy', 'pandas', 'matplotlib', 'scikit-learn', 'scipy', 'joblib', 'threadpoolctl']},
    }
    split_summary = {
        name: {'count': len(indices), 'median_price_kzt': float(frame.loc[indices, 'price'].median()),
               'max_price_kzt': float(frame.loc[indices, 'price'].max()),
               'above_100m_count': int((frame.loc[indices, 'price'] > 100e6).sum())}
        for name, indices in split.items()
    }
    report = {
        'data': {**audit, 'splits': {key: len(value) for key, value in split.items()},
                 'split_price_summary': split_summary, 'numeric_features': NUMERIC, 'categorical_features': CATEGORICAL,
                 'preprocessing': 'All learned preprocessing is fitted on training rows/folds. Four original baselines use unchanged legacy_data.py. Tuned models select OHE min_frequency inside 3-fold train CV; the corrected single ANN and both fixed pools use min_frequency=2.'},
        'eda': eda, 'supervised': supervised, 'ann': ann_report, 'ensemble': ensemble_report,
        'robust_ann': robust_report, 'usage_ann': usage_report,
        'dataset_enrichment': {'directory': 'dataset', **dataset_report}, 'unsupervised': unsupervised,
        'evaluation': {**selection, 'model_rows': rows, 'test_rows': len(y_test),
                       'baseline_improvement_fraction': improvement,
                       'improvement_over_midterm_knn_fraction': 1 - best['test_mae_kzt'] / midterm['test_mae_kzt'],
                       'success_criterion_met': bool(improvement >= .20), 'error_analysis': errors,
                       'historical_test_disclosure': policy,
                       'uncertainty': 'Earlier engineering used known benchmark errors; independent later data is needed to confirm gains. One split and sparse high-price/rare-model segments limit conclusions.'},
        'environment': environment, 'elapsed_seconds': time.perf_counter() - start,
    }
    write_json(OUT / 'report.json', report)
    write_json(OUT / 'environment.json', environment)
    # The only persisted fitted estimator contains all ten ANN checkpoints and RF.
    joblib.dump({
        'selected_model': SELECTED_MODEL, 'estimator': selected, 'preprocessor': None,
        'features': {'enhanced': FEATURES + ['steering', 'log_vehicle_age', 'log_annual_mileage_proxy']},
        'representation': 'mixed', 'currency': 'KZT', 'seed': SEED,
        'feature_derivation': 'Use dataset_features.add_dataset_features then structured_features.add_structured_features before estimator.predict({"enhanced": frame}).',
    }, OUT / 'selected_model.joblib', compress=3)
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(rows))
    width = .38
    ax.bar(x - width / 2, comparison.validation_mae_kzt / 1e6, width, label='Validation', color=ACCENT)
    ax.bar(x + width / 2, comparison.test_mae_kzt / 1e6, width, label='Historical test', color=GREEN)
    ax.set_xticks(x, comparison.model, rotation=25, ha='right')
    ax.set(ylabel='MAE (million KZT)', title='Same-data model comparison')
    ax.legend()
    savefig(fig, 'model_comparison.png')
    print(comparison[['model', 'validation_mae_kzt', 'test_mae_kzt', 'test_rmse_kzt', 'test_r2',
                      'test_median_ape', 'test_within_20pct', 'selected']].round(4).to_string(index=False), flush=True)
    print('Selected:', SELECTED_MODEL, 'test MAE:', round(best['test_mae_kzt']),
          'baseline improvement:', f'{improvement:.1%}', flush=True)
    print('Saved', OUT, 'elapsed seconds', round(report['elapsed_seconds'], 1), flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):
        main()
