## Welcome to the transfyr.ai pipette well challenge!

Your goal is to create a method that predicts the well or wells a pipette is dispensing into in a 96-well plate using the provided video dataset. 

## Input:

This repo provides a tarball of 100 time-synchronized, multi-angle clips of pipetting into a 96-well plate. Labels are provided via an accompanying `labels.json` in the following format: 

```JSON 
[
    {
        "clip_id_FPV": "Plate_10_clip_0029_FPV",
        "clip_id_Topview": "Plate_10_clip_0029_Topview",
        "wells_ground_truth": [
            {
            "well_row": "F",
            "well_column": "4"
            }
        ]
    },
    {
        "clip_id_FPV": "Plate_11_clip_0002_FPV",
        "clip_id_Topview": "Plate_11_clip_0002_Topview",
        "wells_ground_truth": [
        {
            "well_row": "A",
            "well_column": "4"
        }
        ]
    },
    ...
]
```

### Output: 

Your code should take 2 video files as input -- first-person-view (FPV) video path and a topview video path -- and provide a JSON-formatted output like below (this is the same format as the labels, except the key is changed from `wells_ground_truth` to `wells_prediction`). 

```JSON
{
    "clip_id_FPV": "Plate_44_clip_0001_FPV",
    "clip_id_Topview": "Plate_44_clip_0001_Topview",
    "wells_prediction": [
      {
        "well_row": "H",
        "well_column": "1"
      }
    ]
}
```

In the cases where a multi-channel pipette is used, please list all the row-column pairs in the JSON array. 

```JSON
{
    "clip_id_FPV": "Plate_1_clip_0001_FPV",
    "clip_id_Topview": "Plate_1_clip_0001_Topview",
    "wells_prediction": [
        {
            "well_row": "A",
            "well_column": "1"
        },
        {
            "well_row": "A",
            "well_column": "2"
        },
        {
            "well_row": "A",
            "well_column": "3"
        },
        {
            "well_row": "A",
            "well_column": "4"
        },
        {
            "well_row": "A",
            "well_column": "5"
        },
        {
            "well_row": "A",
            "well_column": "6"
        },
        {
            "well_row": "A",
            "well_column": "7"
        },
        {
            "well_row": "A",
            "well_column": "8"
        },
        {
            "well_row": "A",
            "well_column": "9"
        },
        {
            "well_row": "A",
            "well_column": "10"
        },
        {
            "well_row": "A",
            "well_column": "11"
        },
        {
            "well_row": "A",
            "well_column": "12"
        }
    ]
}
```

We will be running your code against a hold-out set to evaluate correctness, so please include a `README.md` about how to run your code. 

## Getting Started 

Please clone this repo, and then create a new private repo within your GitHub user. Replace the remote of your local repo with the newly created private repo’s remote. Share that private repo with a few members of our technical team (GitHub usernames: `kathryn-transfyrai` and `ari-transfyr`).

Additionally, please record a short video 5-10 min that walks through your solution and explains your thought process. Loom or similar services are a great way to record this video! If you have any questions, please don't hesitate to reach out to the Transfyr technical team.

P.S. If you’re stumbling upon this repo and you haven’t had a screening call, shoot us an email at `recruiting [at] transfyr.ai` with your solution! You are awesome and we want to get to know you!

P.P.S. A certain CEO made this sample set for you with water so please forgive the lack of appropriate PPE.
