
from utils import *
df = pd.read_csv('test_data/recordings.csv')

labels = {
    vid_key: {
        'features': df.loc[df.ID.apply(lambda elt: elt == vid_key)].sort_values(by=['Utt Idx'],ascending=True)['Val (0-2)'].to_numpy().reshape((-1,1)),
        # 'features': df.loc[df.ID.apply(lambda elt: elt == vid_key)].sort_values(by=['Utt Idx'],ascending=True)['Act'].to_numpy().reshape((-1,1)),
        'intervals': None
    }
    for vid_key in df.ID.unique()
}
print(labels.keys())
save_pk('test_data/val_labels.pk', labels)
# save_pk('test_data/act_labels.pk', labels)
print(labels)
