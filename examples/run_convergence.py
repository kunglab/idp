import visualize as vz

all_one = [0.228343,
           0.422172,
           0.477255,
           0.52947,
           0.576642,
           0.610463,
           0.580202,
           0.640328,
           0.604134,
           0.613825,
           0.687599,
           0.692247,
           0.721025,
           0.719343,
           0.728738,
           0.73665,
           0.723299,
           0.765328,
           0.742583,
           0.729727,
           0.773438,
           0.784217,
           0.779371,
           0.784909,
           0.722706,
           0.816159,
           0.816159,
           0.811214,
           0.791436,
           0.813192,
           0.788964,
           0.813192,
           0.787282,
           0.812599,
           0.805676,
           0.80449,
           0.80716,
           0.80004,
           0.810324,
           0.798062,
           0.78837,
           0.809237,
           0.779767,
           0.821994,
           0.804786,
           0.782931,
           0.807654,
           0.797666,
           0.80894,
           0.812698,
           0.842267,
           0.853837,
           0.856112,
           0.855716,
           0.853441,
           0.851958,
           0.858287,
           0.856705,
           0.857002,
           0.857793,
           0.854925,
           0.857397,
           0.8571,
           0.855222,
           0.853837,
           0.844838,
           0.836432,
           0.762856,
           0.815862,
           0.82229,
           0.834157,
           0.824862,
           0.826741,
           0.824169,
           0.81962,
           0.851859,
           0.85265,
           0.857002,
           0.858683,
           0.854727,
           0.857694,
           0.859375,
           0.858386,
           0.858881,
           0.860661,
           0.850771,
           0.862638,
           0.859177,
           0.860661,
           0.856309,
           0.858683,
           0.852947,
           0.857991,
           0.856408,
           0.858979,
           0.861254,
           0.860957,
           0.851464,
           0.857199,
           0.8571]

linear = [
    0.336926,
    0.44017,
    0.504055,
    0.547369,
    0.574367,
    0.456685,
    0.549842,
    0.640625,
    0.667227,
    0.696598,
    0.669699,
    0.696796,
    0.720827,
    0.734672,
    0.728046,
    0.752967,
    0.700059,
    0.747824,
    0.74199,
    0.755835,
    0.744264,
    0.762362,
    0.687994,
    0.751483,
    0.785305,
    0.824763,
    0.816653,
    0.801721,
    0.813192,
    0.80983,
    0.798952,
    0.802314,
    0.800138,
    0.81517,
    0.797073,
    0.776998,
    0.773339,
    0.794798,
    0.801523,
    0.816851,
    0.778778,
    0.801226,
    0.805874,
    0.784513,
    0.781448,
    0.794403,
    0.794007,
    0.794106,
    0.740012,
    0.799644,
    0.843453,
    0.849684,
    0.856903,
    0.853145,
    0.852848,
    0.854826,
    0.852156,
    0.856804,
    0.855518,
    0.854826,
    0.850178,
    0.853738,
    0.854331,
    0.852749,
    0.847805,
    0.83485,
    0.8125,
    0.792326,
    0.815071,
    0.805578,
    0.795787,
    0.805676,
    0.823477,
    0.814379,
    0.823774,
    0.851266,
    0.856606,
    0.855419,
    0.858089,
    0.858386,
    0.858089,
    0.858089,
    0.8571,
    0.858485,
    0.848695,
    0.859276,
    0.854727,
    0.854925,
    0.859672,
    0.850574,
    0.855716,
    0.85621,
    0.852354,
    0.854233,
    0.858287,
    0.86165,
    0.853837,
    0.859869,
    0.86165,
    0.859177]

xs_dict = {'all-one': range(100), 'linear': range(100)}
epoch_dict = {'all-one': all_one, 'linear': linear}
colors = [vz.colors.all_one_lg, vz.colors.linear_lg]
vz.plot(xs_dict, epoch_dict, ['all-one', 'linear'], 'loss', colors,
        folder='_figures/', linewidth=1.5, marker='',
        xlabel='Epoch', ylabel='Classification Accuracy', ext='pdf')
