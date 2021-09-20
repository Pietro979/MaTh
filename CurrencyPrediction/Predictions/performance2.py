import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

dates = ['05-04-2021', '05-05-2021', '05-06-2021', '05-07-2021',
     '05-10-2021', '05-11-2021', '05-12-2021' ,'05-13-2021', '05-14-2021',
     '05-17-2021', '05-18-2021', '05-19-2021', '05-20-2021', '05-21-2021',
     '05-24-2021', '05-25-2021', '05-26-2021', '05-27-2021', '05-28-2021',
     '05-31-2021', '06-01-2021' ,'06-02-2021', '06-04-2021', '06-07-2021']

real_values = [4.5539002, 4.5752, 4.5829, 4.5764, 4.5631, 4.5628,
4.5453, 4.5447, 4.5316, 4.5298, 4.5272, 4.5306, 4.5189,
4.4958,   4.4877,   4.4806, 4.4914002, 4.498, 4.4845, 4.4805,
4.4749002, 4.4654, 4.4734, 4.4581]

baseline_predictedValues = [4.5654,    4.5539002, 4.5752 ,   4.5829 ,   4.5764,    4.5631 ,   4.5628,
 4.5453  ,  4.5447  ,  4.5316  ,  4.5298 ,   4.5272  ,  4.5306  ,  4.5189,
 4.4958 ,   4.4877  ,  4.4806 ,   4.4914002, 4.498  ,   4.4845 ,   4.4805,
 4.4749002, 4.4654  ,  4.4734   ]



linear_predictedValues = [4.5592504, 4.5479546, 4.5688763, 4.5764394 ,4.570055,  4.556991 , 4.5566964,
 4.5395074 ,4.538918 , 4.5260506, 4.5242825, 4.5217285, 4.5250683, 4.513576,
 4.490886,  4.48293 ,  4.475956,  4.486564,  4.493047 , 4.479787 , 4.4758577,
 4.4703574, 4.4610257 ,4.468884 ]

dense_predictedValues = [4.5580354, 4.5468535, 4.5675645 ,4.575052,  4.5687313 ,4.555799 , 4.555507,
 4.53849,   4.537906,  4.525165 , 4.523414 , 4.5208855, 4.5241923, 4.5128126,
 4.490345 , 4.482467 , 4.4755616, 4.486066,  4.492485 , 4.479355 , 4.4754643,
 4.4700174 ,4.4607778, 4.4685583]

conv_predictedValues = [4.5493283, 4.5372677, 4.553983,  4.566226,  4.5600867 ,4.5467596 ,4.54448,
 4.5315285, 4.527237 , 4.518483,  4.514188 , 4.5133886, 4.515836 , 4.507193,
 4.4847813 ,4.4745154 ,4.4699173, 4.478529 , 4.4871335, 4.4760413 ,4.468835,
 4.4650908, 4.4564333, 4.4618306]

recurr_predictedValues = [4.4924765, 4.5184946, 4.5352597, 4.5399046, 4.5357957, 4.528134 , 4.5285816,
 4.51978,   4.5192404, 4.512121 , 4.5106864, 4.5097322, 4.51232 ,  4.5048676,
 4.487223 , 4.479813 , 4.4754806, 4.4845653, 4.490658,  4.479389,  4.474063,
 4.470009 , 4.462457,  4.4689403]


mult_baseline_predicteDValues = [4.5654, 4.5654, 4.5654, 4.5654, 4.5654, 4.5654, 4.5654, 4.5654, 4.5654, 4.5654,
 4.5654, 4.5654, 4.5654, 4.5654, 4.5654, 4.5654, 4.5654, 4.5654, 4.5654, 4.5654,
 4.5654, 4.5654 ,4.5654 ,4.5654]

mult_repeatBaseline_predicteDValues = [4.656 ,    4.6519 ,   4.6603003, 4.6239,    4.5933,    4.6041,    4.5897,
 4.5581 ,   4.5414  ,  4.533 ,    4.5627003, 4.5552 ,   4.5546002, 4.5481,
 4.5474 ,   4.5541 ,   4.5567 ,   4.5562,    4.5649,    4.5565,    4.5613003,
 4.5811,    4.5782 ,   4.5654   ]

mult_linear_predicteDValues = [4.560283,  4.555527,  4.5504274 ,4.5455985 ,4.5412874, 4.5370073 ,4.5323987,
 4.527719,  4.523402,  4.5187583, 4.5146885, 4.5107393 ,4.5069013 ,4.5032663,
 4.498864,  4.494129,  4.4896116, 4.4851513, 4.4804525, 4.4764056, 4.4730434,
 4.469852,  4.467129,  4.464008 ]

mult_dense_predicteDValues = [4.5331273, 4.5188675, 4.5140142, 4.507621 , 4.509092 , 4.50655,  4.507294,
 4.505864,  4.5032 ,   4.4942775 ,4.492792 , 4.4937596, 4.4820437 ,4.4804635,
 4.4711304, 4.468155,  4.4603534, 4.4625583, 4.4593854 ,4.452311,  4.446305,
 4.444423,  4.4436154, 4.441632 ]

mult_conv_predicteDValues = [4.5510044, 4.5402937, 4.5214972, 4.520947,  4.5192704, 4.5149145, 4.513035,
 4.5082264, 4.497893,  4.49113,   4.4886994, 4.486342,  4.4809613, 4.47623,
 4.4689336, 4.4655204, 4.4629664, 4.4609017, 4.4561596, 4.454909,  4.4520407,
 4.4479876, 4.444741 , 4.4440536]

mult_recurr_predicteDValues = [4.4801426, 4.4818354, 4.481517,  4.4849634, 4.486214,  4.487223,  4.4886904,
 4.487291 , 4.4866786, 4.4850483, 4.4829836, 4.4794664 ,4.4741807 ,4.4685416,
 4.4615445 ,4.4543676 ,4.446567,  4.4391093, 4.431588 , 4.424053 , 4.4182034,
 4.4132433 ,4.408011 , 4.404955 ]


def compare_errors(real_values = None, predicted_values = None):
    diff_sum = []
    diffs = []
    for i in range(len(real_values)):
        # print("Real value: ", real_values[i], "Predicted value: ", predicted_values[i])
        difference = abs(real_values[i] - predicted_values[i])
        if i >0:
            diff_sum.append(diff_sum[i-1] + difference)
        else:
            diff_sum.append(difference)
        diffs.append(predicted_values[i]-real_values[i])
        # diff_sum = np.array(diff_sum)
        # diffs = np.array(diffs)
    return diff_sum, diffs


diff_sum_base, diffs_base = compare_errors(real_values, baseline_predictedValues)
diff_sum_linear, diffs_linear = compare_errors(real_values, linear_predictedValues)
diff_sum_dense, diffs_dense = compare_errors(real_values, dense_predictedValues)
diff_sum_conv, diffs_conv = compare_errors(real_values, conv_predictedValues)
diff_sum_recurr, diffs_recurr = compare_errors(real_values, recurr_predictedValues)

diff_sum_mult_base, diffs_mult_base = compare_errors(real_values, mult_baseline_predicteDValues)
diff_sum_mult_repeat, diffs_mult_repeat = compare_errors(real_values, mult_repeatBaseline_predicteDValues)
diff_sum_mult_linear, diffs_mult_linear = compare_errors(real_values, mult_linear_predicteDValues)
diff_sum_mult_dense, diffs_mult_dense = compare_errors(real_values, mult_dense_predicteDValues)
diff_sum_mult_conv, diffs_mult_conv = compare_errors(real_values, mult_conv_predicteDValues)
diff_sum_mult_recurr, diffs_mult_recurr = compare_errors(real_values, mult_recurr_predicteDValues)


####################################################################
# fig = plt.figure()
# ax = fig.add_axes([0,0,1,1])
# ax.bar(dates,diffs_base)
# plt.show()

# plt.bar(x = range(len(dates)),
#         height=diffs_mult_recurr)
# axis = plt.gca()
# axis.set_xticks(range(len(dates)))
# _ = axis.set_xticklabels(dates, rotation=90)
# axis.set_xlabel('Dates')
# axis.set_ylabel('Absolute error')
# plt.show()

# plt.bar(x = range(len(dates)),
#         height=diffs_mult_conv)
# axis = plt.gca()
# axis.set_xticks(range(len(diffs_mult_recurr)))
# _ = axis.set_xticklabels(dates, rotation=90)
# axis.set_xlabel('Dates')
# axis.set_ylabel('Absolute error')
# plt.show()


# fig = plt.figure()
# plt.plot(dates, diff_sum_base, color='green')
# plt.scatter(dates, diff_sum_base,  # bez ostatniego
#                         edgecolors='k', label='Baseline', c='green', s=64)
#
# plt.plot(dates, diff_sum_linear, color='red')
# plt.scatter(dates, diff_sum_linear,  # bez ostatniego
#                         edgecolors='k', label='Linear', c='red', s=64)
#
# plt.plot(dates, diff_sum_dense, color='blue')
# plt.scatter(dates, diff_sum_dense,  # bez ostatniego
#                         edgecolors='k', label='Dense', c='blue', s=64)
#
# plt.plot(dates, diff_sum_conv, color='yellow')
# plt.scatter(dates, diff_sum_conv,  # bez ostatniego
#                         edgecolors='k', label='CNN', c='yellow', s=64)
#
# plt.plot(dates, diff_sum_recurr, color='purple')
# plt.scatter(dates, diff_sum_recurr,  # bez ostatniego
#                         edgecolors='k', label='RNN', c='purple', s=64)
# ax = plt.gca()
# temp = ax.xaxis.get_ticklabels()
# temp = list(set(temp) - set(temp[::5]))
# plt.xlabel("Dates")
# plt.ylabel("Sum of errors")
# for label in temp:
#     label.set_visible(False)
# ax.legend()
# plt.show()
# ####################################################################
#
# fig = plt.figure()
# ax = fig.add_axes([0,0,1,1])
# ax.bar(dates,diffs_base)
# plt.show()

# plt.bar(x = range(len(dates)),
#         height=diffs_base)
# axis = plt.gca()
# axis.set_xticks(range(len(diffs_base)))
# _ = axis.set_xticklabels(dates, rotation=90)


fig = plt.figure()
plt.plot(dates, diff_sum_mult_base, color='green')
plt.scatter(dates, diff_sum_mult_base,  # bez ostatniego
                        edgecolors='k', label='Baseline', c='green', s=64)

plt.plot(dates, diff_sum_mult_repeat, color='red')
plt.scatter(dates, diff_sum_mult_repeat,  # bez ostatniego
                        edgecolors='k', label='Repeat baseline', c='red', s=64)

plt.plot(dates, diff_sum_mult_linear, color='blue')
plt.scatter(dates, diff_sum_mult_linear,  # bez ostatniego
                        edgecolors='k', label='Linear', c='blue', s=64)

plt.plot(dates, diff_sum_mult_dense, color='yellow')
plt.scatter(dates, diff_sum_mult_dense,  # bez ostatniego
                        edgecolors='k', label='Dense', c='yellow', s=64)

plt.plot(dates, diff_sum_mult_conv, color='purple')
plt.scatter(dates, diff_sum_mult_conv,  # bez ostatniego
                        edgecolors='k', label='CNN', c='purple', s=64)

plt.plot(dates, diff_sum_mult_recurr, color='pink')
plt.scatter(dates, diff_sum_mult_recurr,  # bez ostatniego
                        edgecolors='k', label='RNN', c='pink', s=64)

ax = plt.gca()
temp = ax.xaxis.get_ticklabels()
temp = list(set(temp) - set(temp[::5]))
plt.xlabel("Dates")
plt.ylabel("Sum of errors")
for label in temp:
    label.set_visible(False)

ax.legend()
plt.show()
#
#
#
#
def max_diff(diffs):
    diffs_abs = []
    for i in diffs:
        diffs_abs.append(abs(i))
    maximum = max(diffs_abs)
    minimum = min(diffs_abs)
    return maximum, minimum

# maxs = []
# maxs.append(max_diff(diffs_base)[0])
# maxs.append(max_diff(diffs_linear)[0])
# maxs.append(max_diff(diffs_dense)[0])
# maxs.append(max_diff(diffs_conv)[0])
# maxs.append(max_diff(diffs_recurr)[0])
#
# print("wartość maksymalna: ",max(maxs)," indeks: ", maxs.index(max(maxs)) +1)
#
# mins = []
# mins.append(max_diff(diffs_base)[1])
# mins.append(max_diff(diffs_linear)[1])
# mins.append(max_diff(diffs_dense)[1])
# mins.append(max_diff(diffs_conv)[1])
# mins.append(max_diff(diffs_recurr)[1])
#
# print("wartość minimalna: ",min(mins)," indeks: ", mins.index(min(mins))+1)


maxs = []
maxs.append(max_diff(diffs_mult_base)[0])
maxs.append(max_diff(diffs_mult_repeat)[0])
maxs.append(max_diff(diffs_mult_linear)[0])
maxs.append(max_diff(diffs_mult_dense)[0])
maxs.append(max_diff(diffs_mult_conv)[0])
maxs.append(max_diff(diffs_mult_recurr)[0])

print("wartość maksymalna: ",max(maxs)," indeks: ", maxs.index(max(maxs)) +1)

mins = []
mins.append(max_diff(diffs_mult_base)[1])
mins.append(max_diff(diffs_mult_repeat)[1])
mins.append(max_diff(diffs_mult_linear)[1])
mins.append(max_diff(diffs_mult_dense)[1])
mins.append(max_diff(diffs_mult_conv)[1])
mins.append(max_diff(diffs_mult_recurr)[1])

print("wartość minimalna: ",min(mins)," indeks: ", mins.index(min(mins))+1)