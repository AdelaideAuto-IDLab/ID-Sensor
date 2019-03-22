# ID-Sensor dataset

Dataset for the ID-Sensor paper. The total data set contains 234,764 tag readings which included 109,141 _in-bed_ and 125,623 _out-of-bed_ related tag readings. Sampling rate was generally between 5-20 reads per second. The data set contains 70 bed-exit events (i.e. transitions from _in-bed_ to _out-of-bed_). See section III of the paper for additional details.

The dataset consists of two files:

* `XL.csv` contains the tag readings:
    - `time` - The time (in seconds) since the first reading in a session.
    - The fields `acc_frontal`, `acc_vertical`, `acc_lateral` correspond to tag 1 and are unused
    - `antenna` - Which antenna the reading corresponded to.
    - `rssi` - The signal strength of the received signal (in dB).
    - `tag_id` - Which tag the reading corresponded to. Note tag 1 corresponds to data from a different sensor, and is unused.
    - `phase` - The phase of the received signal (in radians)
    - `frequency` - The frequency of the received signal (in MHz)

* `YL.csv` contains a list of all labels associated with each attribute:
    - 0 = separation
    - 1 = sit_on_bed
    - 2 = sit_on_chair
    - 3 = lying_on_bed
    - 4 = walk
    - 5 = stand_up