
from utils import *
from interpret import get_df

if __name__ == '__main__':
    hash='06819'
    df = get_df(f'results/{hash}/csv_results.csv')
    df['acc'] = df['test_acc'].map(lambda elt: npr(ar(json.loads(elt)).mean()))
    print('Results of grid search: ')
    print(df[['acc','cross_utterance', 'modality']].sort_values(by=['cross_utterance', 'modality']))
