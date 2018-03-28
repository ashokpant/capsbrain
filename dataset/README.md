## Dataset preparation utilities

### smallNORB Dataset

```bash
cd data
mkdir smallNORB
cd smallNORB
chmod +x download.sh;./download.sh
```
Download will take a few minutes or seconds, the whole data set is about 900MB unzipped. They are placed under the ```smallNORB``` folder.


Generate TFRecord files for both train and test dataset.
```python
python smallNORB.py tfrecord
```