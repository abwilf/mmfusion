
from utils import *
df = pd.read_csv('test_data/recordings.csv')

labels = {
    vid_key: {
        'features': df.loc[df.ID.apply(lambda elt: elt == vid_key)].sort_values(by=['Utt Idx'],ascending=True)['Val (0-2)'].to_numpy().reshape((-1,1)),
        'intervals': None
    }
    for vid_key in df.ID.unique()
}
save_pk('test_data/labels.pk', labels)
print(labels)
