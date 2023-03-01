
Dataloader for large sequnce data
* [Done] when your data localy
* [TODO] when your data in HDFS

Before you start - check webdataset github and pytorch-lifestream github, there more universary way to work with large data.

Prepare your data as in example notebook:
1) All sequence columns -> list[x1,x2,x3]. In ascending order by your datetime.
Before you have many row for ID, now you have one row for ID, because sequence became list[].
2) Save data to parquets file. Keep in mind - the number of resulting data files must be divided without a remainder by the number of num_workers that you plan to use in the DataLoader.

Than modify dataloader.py (SequenceCollator and get_dataloader func) for your specific task and data.

It wotks for Pytorch framework.
For now it required webdataset==0.1.62

