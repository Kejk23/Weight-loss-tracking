import numpy as np
import pandas as pd
from scipy.odr import ODR, Model, RealData
import matplotlib.pyplot as plt
from tabulate import tabulate
import string

from uncertainties import ufloat
import uncertainties.unumpy as unp
import uncertainties as u

# More scientific formatting for plots
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"], 
})

# Custom ShorthandFormatter class for uncertainties package
class ShorthandFormatter(string.Formatter):
    def format_field(self, value, format_spec):
        if isinstance(value, u.UFloat):
            num, exponent = "{:.3e}".format(value.std_dev).split('e')  
            num = num.replace('.', '')  
            num = num.ljust(3, '0')
            first_digit = int(num[0])
            second_digit = int(num[1])
            third_digit = int(num[2])
 
            if first_digit == 1:
                if second_digit == 0 and third_digit >= 5:
                    return value.format(format_spec+'.2uS')
                elif second_digit == 9 and third_digit < 5:
                    return value.format(format_spec+'.2uS')
                elif second_digit in range(1,9):
                    return value.format(format_spec+'.2uS')
                else:
                    return value.format(format_spec+'.1uS') 
            else:
                return value.format(format_spec+'.1uS') 
        else:
            return super(ShorthandFormatter, self).format_field(value, format_spec)
        
# Initialize the formatter
frmtr = ShorthandFormatter()
        
# Data cleaning function
def clean_data(x, y, exclude_indices=None):
    # Remove specified indices
    if exclude_indices is not None:
        x = np.delete(x, exclude_indices)
        y = np.delete(y, exclude_indices)
 
    # Adjust x and y for UFloat type
    x2 = x if not isinstance(x[0], u.UFloat) else unp.nominal_values(x)
    y2 = y if not isinstance(y[0], u.UFloat) else unp.nominal_values(y)
 
    # Remove NaN and Inf values
    invalid_indices_x = np.argwhere(np.isnan(x2) | np.isinf(x2))
    invalid_indices_y = np.argwhere(np.isnan(y2) | np.isinf(y2))
    invalid_indices = np.unique(np.concatenate((invalid_indices_x, invalid_indices_y), axis=None))
    x = np.delete(x, invalid_indices)
    y = np.delete(y, invalid_indices)
 
    return x, y

# ODR fitting function
def odr_params(x, y, fitted_func, initial_params=None, exclude_indices=None):
    x, y = clean_data(x, y, exclude_indices)
 
    # Replace initial parameter guesses with ones if not provided
    if initial_params is None:
        initial_params = np.ones(fitted_func.__code__.co_argcount - 1)
 
    # Adjust x and y for UFloat type
    if isinstance(x[0], u.UFloat):
        x_values = unp.nominal_values(x)
        x_errors = unp.std_devs(x)
    else:
        x_values = x
        x_errors = None
    if isinstance(y[0], u.UFloat):
        y_values = unp.nominal_values(y)
        y_errors = unp.std_devs(y)
    else:
        y_values = y
        y_errors = None
 
    # Create model and RealData object for ODR fit
    def model_func(B, x):
        return fitted_func(x, *B)
    
    model = Model(model_func)
    data = RealData(x_values, y_values, sx=x_errors, sy=y_errors)
 
    # Perform ODR fit
    result = ODR(data, model, beta0=initial_params).run()
    params = [u.ufloat(nominal, uncertainty) for nominal, uncertainty in zip(result.beta, result.sd_beta)]
    return params

# Define the fitting function
def weight_loss_model(x, a, b):
    return a * b**((x-1) / 7)
# We should aim to loose between 0.5% and 1% of our body weight per week
target_b_min = 0.99
target_b_max = 0.995

# Load the weight loss data from the Excel file
df = pd.read_excel('data.xlsx')
# Calculate the "days" array based on the date difference
start_date = df['Date'].min()
days = np.array([(date - start_date).days + 1 for date in df['Date']])
# Weights array is directly the 'Weight' column
weights = df['Weight'].values

# Standard deviation of a uniform distribution
# I also use scale with delta_max = 0.05 sometimes
# However, this should be ok in most cases
delta_max = 0.1
scale_std = delta_max/ np.sqrt(12)

# Now lets change the numpy arrays to unumpy arrays
days_unp = unp.uarray(days, 1/24) # I weight myself +- 1 hour from the same time in the morning
weights_unp = unp.uarray(weights, scale_std)

# Perform ODR fitting (too complicated, I know, but why not? It's a physics project after all)
[a_ufloat, b_ufloat] = odr_params(days_unp, weights_unp, weight_loss_model, initial_params=[86, 0.99])
print(frmtr.format("Weekly weight loss coeff = {} ", b_ufloat))
# Check if the weight loss is too fast, too slow, or on track
if b_ufloat < target_b_min:
    print("You are losing weight too fast!")
elif b_ufloat > target_b_max:
    print("You are losing weight too slow!")
else:
    print("You are on track!")
print(frmtr.format("Estimated starting weight = {} kg", a_ufloat))
print(frmtr.format("Estimated current weight = {} kg", a_ufloat*b_ufloat**((max(days)-1)/7)))
print(frmtr.format("Estimated weight lost = {} kg", a_ufloat*(1-b_ufloat**((max(days)-1)/7))))

# Plotting
plt.figure(figsize=(10, 6))
plt.errorbar(days, weights, yerr=weights_unp[0].s, xerr=days_unp[0].s, fmt='o', label='Data', color='red') 
# We want to make some space for the target range in future days
days_plus_seven = np.concatenate((days, np.arange(days[-1] + 1, days[-1] + 8)))
plt.plot(days, weight_loss_model(days, *unp.nominal_values([a_ufloat, b_ufloat])), label='Fitted Curve', color='red')
plt.fill_between(days_plus_seven,  
                 weight_loss_model(days_plus_seven, a_ufloat.nominal_value - a_ufloat.std_dev, b_ufloat.nominal_value - b_ufloat.std_dev), 
                 weight_loss_model(days_plus_seven, a_ufloat.nominal_value + a_ufloat.std_dev, b_ufloat.nominal_value + b_ufloat.std_dev), 
                 color='red', alpha=0.3, label='Uncertainty')

# Plot target weight loss lines
plt.fill_between(days_plus_seven,  
                 weight_loss_model(days_plus_seven, a_ufloat.nominal_value - a_ufloat.std_dev, target_b_min), 
                 weight_loss_model(days_plus_seven, a_ufloat.nominal_value + a_ufloat.std_dev, target_b_max), 
                 color='blue', alpha=0.3, label='Target Range') 

plt.xlabel('Days')
plt.ylabel('Weight (kg)')
plt.title('Weight Loss Journey')
plt.legend()
plt.grid(True)

# Human mind is not very good at understanding trends 
# We much rather prefer to think in termns of actual weights and kgs
# However, this is problematic since weight fluctuates a lot on a daily basis and doesn't acrtually reflect the ammount of fat
# Most people are affraid to step on the scale, because the are emotionally attached to the number they see
# This approach can help people undrestand, that each daily measuremnt is just a drop in the ocean and doesn't really matter
# In my opinion I think this is better then not weighting yourself at all, even though it can work for some people
# For me personaally, I would never loose weight without quantifiable feedback and accountability
# This approach helps me stay motivated and prevents me from over and underestimating my progress, hitting plateaus, etc.

# We can utilize human obsession with numbers to our advantage though
# Instead of daily measuremnts, we can use weekly median weight as a unit of "how much do I weight"
# This way we can still have a number to obsess about, but it will be more stable and less prone to fluctuations 

# Let's calculate the median value for each week
# and make target ca
weeks_completed = max(days)//7
weekly_medians = []
for week in range(1, weeks_completed + 1):
    week_start = (week - 1) * 7 + 1
    week_end = week * 7
    week_weights = weights[(days >= week_start) & (days <= week_end)]
    weekly_medians.append(np.median(week_weights))
this_week_median = np.median(weights[days >= weeks_completed * 7 + 1])
print()
print("Weeks completed:", weeks_completed)
print("Weekly medians (kg):", weekly_medians)
print(f"This week's median weight = {this_week_median} kg")
# Even on weekly basis, it's best to not loose weight too fast or too slow
target_min_weight = round(weekly_medians[-1] * 0.99, 1)
target_max_weight = round(weekly_medians[-1] * 0.995, 1)
print(f"This week's target min = {target_min_weight} kg")
print(f"This week's target max = {target_max_weight} kg")
print()

# Long term, the ranges are much broader, stil it's best to calculate them I think
total_weeks = 12
weeks = list(range(1, total_weeks+1))
target_min_weights = [weekly_medians[0] * (target_b_min ** (week-1)) for week in weeks]
target_max_weights = [weekly_medians[0] * (target_b_max ** (week-1)) for week in weeks]
table_data_nominal = []
for week, min_weight, max_weight in zip(weeks, target_min_weights, target_max_weights):
    table_data_nominal.append([
        week,
        round(min_weight, 1),
        round(max_weight, 1)
    ])
# Display the results using tabulate
headers = ["Week", "Target Min Weight (kg)", "Target Max Weight (kg)"]
print("Target long term weight loss ranges:")
print(tabulate(table_data_nominal, headers=headers, floatfmt=".1f")) # it's easier to do the rounding here

plt.savefig('weight_loss.png', dpi=300)

# Keep the console window open
print()
input("Press Enter to exit...")