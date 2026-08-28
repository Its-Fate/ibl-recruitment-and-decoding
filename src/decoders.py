import pandas as pd
from preprocessing import preprocess_region, make_window_X
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score


def decode_logistic(X, y):

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.05, max_iter=2000, class_weight='balanced')
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    return cross_val_score(model, X, y, cv=cv, scoring="balanced_accuracy").mean()

def decode(meta, centers, label, win):
    regions = meta.clusters['acronym'].unique()
    regions = [r for r in regions if r not in ['root', 'void']]

    contrast_results = []

    for reg in regions:

        region_data = preprocess_region(reg, meta)

        for data in region_data:
            psth = data['psth']
            if label == 'contrast':
                y = data['contrast']
            elif label == 'side':
                y = data['side']
            else:
                raise ValueError('Label should be either "contrast" or "side"')

            for center in centers:
                X = make_window_X(psth, meta.times, center, win=win)

                score = decode_logistic(X, y)
                contrast_results.append({
                    'region': reg,
                    'eid': data['eid'],
                    'center': center,
                    'score': score
                })



        print(f"{reg}: done!")

    return pd.DataFrame(contrast_results)