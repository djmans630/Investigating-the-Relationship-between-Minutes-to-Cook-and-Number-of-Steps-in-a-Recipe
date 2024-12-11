# Investigating the Relationship between Minutes to Cook and Number of Steps in a Recipe
### By Daniel Mansperger
---




## Introduction


The data I will use for my project is a set of recipes and several characteristics  
about each one. One example of a recipe can be found here: 
[Chickpea and Fresh Tomato Toss Recipe - Food.com.](https://www.food.com/recipe/chickpea-and-fresh-tomato-toss-51631) 
The dataset itself contains tens of thousands of recipes.  

I begin with two dataframes, which I call `recipes` and `interactions`. `recipes`  
contains information about the characteristics of each recipe, and `interactions`  
contains ratings and reviews for each recipe. Luckily for me, they each contain a  
column for recipe ID’s, which are unique identifiers. Using a left merge (where  
`recipes` is the left dataframe), I merge the information of the two datasets and  
ensure that there is information about all recipes available, even those that  
don’t have review information. I call this dataframe that results from the merge  
`df`, and it will be foundational moving forward.  

Using this data, I will attempt to answer this question: What is the relationship  
between cooking time in minutes and the number of steps and ingredients a recipe  
has? Moving forward, I will develop a prediction model that attempts to predict  
the number of minutes a recipe will take to cook, based on these features and  
more. I believe this question is worth investigating because the number of  
ingredients and steps are two of the most important characteristics when deciding  
on a recipe, as is the time it will take to complete. Based on intuition, I  
believe there is a relationship between these characteristics, and because of  
that I may be able to determine this important information reliably, and be able  
to evaluate the truth behind the specifics of a recipe’s claimed time to complete.  
Should this be the case, potential chefs can have more information at their  
disposal to help them decide on a recipe that meets their desired time criteria.  

Before any EDA or cleaning, `df` has 234,429 rows (recipes, with some duplicates  
for multiple people reviewing the same recipe) and 15 columns. Some of these  
columns will be very important for analysis moving forward. The three that I have  
mentioned before are `n_steps`, `n_ingredients`, and `minutes`. `n_steps` is a  
column of integers representing the number of steps in a recipe. Similarly,  
`n_ingredients` is the number of ingredients in a recipe. These are based off of  
the `steps` and `ingredients` columns, respectively, each of which initially  
contain lists of strings of the steps and ingredients of recipes, also  
respectively, though initially the lists in both columns are represented as  
strings where the contents are lists of strings. The `minutes` column is also  
composed of integers, each of which represents the expected number of minutes it  
will take to complete a recipe. Other columns that may be of interest are  
`recipe_id` (containing integers that are unique for each recipe), `name`  
(containing strings representing recipe names), `tags` (containing lists of short  
string tags of information about the recipe, similar to hashtags), `nutrition`  
(containing strings that evaluate to lists of nutrition information about the  
recipe, with each list having the form “[calories (#), total fat (PDV), sugar  
(PDV), sodium (PDV), protein (PDV), saturated fat (PDV), carbohydrates (PDV)]”,  
where PDV stands for “percentage of daily value”), `description` (containing  
strings of the original poster’s description of the recipe), and `rating`  
(containing floats representing a star rating from 1.0 to 5.0). Other columns  
that are not as important to my analysis but are still important in the dataset  
are `date` (for dates), `user_id` (for unique user identification), `avg_rating`  
(a column I added that contains the average ratings of each recipe), and `review`  
(containing someone’s string review of a recipe).





## Data Cleaning and Exploratory Data Analysis
To get a better understanding of the data, I first did a little Exploratory Data
Analysis (EDA). The first thing I checked was the number of missing values in 
each column. There was one recipe missing a name, 114 missing descriptions, 
15,036 missing ratings, 58 missing reviews, and 2,777 missing average ratings. 
All other columns had no missing values. One conclusion I draw from this is 
that some ratings are missing where average ratings exist, which means that 
some people just don’t provide ratings, even on what they may leave a review on 
considering the significantly less missing values in the review column. It 
should be noted that before I looked at any of these missing values, I replaced 
all of the ratings of 0.0 with `np.nan`. The reason for this is that you can 
only provide a rating between 1 and 5 stars for a recipe, so any rating of 0.0 
in the dataset is actually missing and should be treated as such. However, this 
was not the only column that required cleaning, and in my EDA and cleaning, I 
addressed these issues.

The date column was originally composed of strings, and to more easily work 
with the datetime aspects of these values, I opted to convert each entry to 
`pd.datetime` objects, as the form of each date string was consistent. This 
column now contains datetime objects with a day, month, and year of user 
interaction with a recipe.

The nutrition column was one of those composed of strings that contain lists, 
so I applied `eval` to extract those lists. Afterwards, I thought it would be 
easier to work with proportions of daily value rather than percentages, so I 
converted all of the PDV values (all except calories) in accordance. Each 
element of this column is now a list of the form `[calories, total fat, sugar, 
sodium, protein, saturated fat, carbohydrates]`, where all entries except for 
calories are proportions of daily value.

The steps and ingredients columns were similarly composed of strings containing 
lists, so I once again applied `eval` to both of them. Since their lists were 
only composed of text details, this was all I needed to do for these columns, 
which now contain lists of strings representing steps and ingredients, 
respectively.

I then came to the minutes column, which required special consideration. I knew 
I was going to use this column for analysis and prediction moving forward, so I 
needed to make sure I had good information. I noticed that many values were 
actually very high in the minutes column, even going as far as over one million 
minutes! I wanted to limit the recipes I analyzed to those that take less than 
a day to cook. Many recipes are much longer than that, but generally they 
represent recipes for ingredients or require special aging for extended periods 
of time. To accomplish this, of course I would limit the recipes to those that 
take less than a day. But believing I could narrow it down more, I did a little 
outside research. According to [Smoking Times and Temperatures Chart for Beef, 
Pork & Poultry](https://www.smoking-meat.com/smoking-times-and-temperatures-
chart), recipes that can be completed in one day can take anywhere from less 
than an hour to up to 20 hours! Taking the average of these times listed, I get 
about 5 hours, which is around what I wanted because I wanted to filter to only 
include the cooking times of “standing” recipes, or those that could be 
completed in one standing without much downtime. So, I took this to be my 
"minute-limit" on cooking times. 5 hours in minutes is 300 minutes, and as such 
I filtered my dataframe to only include those rows where minutes to cook were 
less than or equal to 300. One other consideration I had to help me filter for 
standing recipes was to use the columns involving text and filter by certain 
keywords. I chose to avoid doing this because some appetizers/snacks certainly 
should be included due to longer cooking times, and in other instances names 
and reviews may include words like "drink" or "wine" because the recipe pairs 
well with these things. As such, I will only filter by time in minutes.

After these cleaning steps are taken, here is a look at the head of my cleaned 
dataframe (with only the columns cleaned/filtered):

| rating | date                | nutrition       | steps                   | ingredients          | minutes |
|-------:|:--------------------|:----------------|:------------------------|:---------------------|--------:|
|      4 | 2008-10-27 00:00:00 | [138.4, 0.1,...]| ['heat...', 'line...']   | ['bittersweet...',...]|      40 |
|      5 | 2011-04-11 00:00:00 | [595.1, 0.46,...]| ['preheat...', 'sift...']| ['white sugar',...]  |      45 |
|      5 | 2008-05-30 00:00:00 | [194.8, 0.2,...]| ['preheat...', 'mix...'] | ['frozen broccoli',...|      40 |
|      5 | 2008-05-30 00:00:00 | [194.8, 0.2,...]| ['preheat...', 'mix...'] | ['frozen broccoli',...|      40 |
|      5 | 2008-05-30 00:00:00 | [194.8, 0.2,...]| ['preheat...', 'mix...'] | ['frozen broccoli',...|      40 |


Below is a graph showing the distribution of cooking times in minutes after the 
filtering. Each bin on the x-axis represents a range of 15 minutes, and the 
y-axis represents the number of UNIQUE recipes with cooking times in each 
range. 

To achieve this, I removed duplicate recipes by dropping duplicate `recipe_id` 
values, keeping only the first occurrence. This approach is valid here since 
the only information lost includes some individual reviews, dates, user IDs, 
and ratings, none of which are relevant for this analysis. All other columns, 
including recipe-specific characteristics, are preserved.

<iframe
  src="assets/minutes_distribution.html"
  width="800"
  height="300"
  frameborder="0"
></iframe>

From this graph, we can see that most of the recipes have minute counts below 
100, and even more are below 50. This passes my sanity checks, as while these 
counts are high, there are still many recipes above these thresholds.

It will also be important to investigate the distribution of `n_steps` and 
`n_ingredients`, due to my question of investigating the relationship between 
these columns and the cooking time in minutes. Starting with `n_steps`, here is 
a graph of the distribution of the values. Each bin is 5 steps wide, and notice 
how wide the graph is. That is because some (very few) recipes have step counts 
above even 60! However, once again, most of the step counts are lower, between 
5 and 15, which tracks.


<iframe
  src="assets/n_steps_distribution.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>


Similarly, let’s check the distribution of `n_ingredients`. This graph takes on 
a more normal shape, with the bulk of recipes having between 6 and 11 
ingredients.


<iframe
  src="assets/n_ingredients_distribution.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>


With both of these graphs, and for the rest of the plots in this EDA section, 
it is important to keep in mind that we are looking at the subset of unique 
recipes that take <= 300 minutes to cook.

With these distributions in mind, and considering all of the values in these 
columns are quantitative, I then moved onto plotting scatter plots for each 
combination of the three columns to see if there was any obvious trend. Sadly, 
the scatter plots were a bit difficult to interpret.

Below is the scatter plot for number of steps (x-axis) and minutes to cook 
(y-axis). I chose these axes because I am taking minutes to be a more dependent 
variable. Unfortunately, no clear trend seems to be present.


<iframe
  src="assets/steps_minutes_scatter.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>


I followed a similar procedure for plotting n_ingredients vs. minutes, once 
again using minutes as the dependent variable and not seeing a clear trend. See
the graph below:


<iframe
  src="assets/ingredients_minutes_scatter.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>


Disappointed with the lack of a clear trend, I went on to plot the relationship 
between the number of steps and the number of ingredients. While this also 
doesn’t show too clear of a trend in the way I was hoping for, it got me 
thinking that each step/ingredient count has a lot of variability in the other 
count. It may be the case that I need to aggregate or look at the features 
together to see something.


<iframe
  src="assets/steps_ingredients_scatter.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

After I pivoted with `n_ingredients` and `n_steps` as the columns and index, 
with the mean minutes as the values, I came to see the trend I was expecting 
pretty clearly. I won’t show that exact table, since there were many missing 
values, but by binning the ingredient and step counts and repeating the 
process, the result is this:

							Ingredients block

| steps_block   |     1-5 |     5-6 |     6-7 |     7-8 |     8-9 |    9-10 |   10-11 |   11-12 |   12-14 |   14-37 |
|:--------------|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|
| 1-4           | 15.5538 | 22.9704 | 28.7969 | 32.0918 | 34.6261 | 40.4548 | 42.5841 | 46.0292 | 52.0532 | 64.9861 |
| 4-5           | 28.6936 | 32.8758 | 34.6355 | 36.0757 | 38.5944 | 46.1458 | 43.8667 | 43.6034 | 52.4017 | 62.0479 |
| 5-6           | 33.3199 | 37.4881 | 37.2865 | 40.6751 | 41.0836 | 46.5852 | 45.9646 | 50.4211 | 54.0708 | 57.8128 |
| 6-8           | 35.5095 | 38.7829 | 39.994  | 42.0173 | 44.4738 | 44.0206 | 45.7723 | 49.8715 | 54.5674 | 60.1082 |
| 8-9           | 37.2272 | 41.0145 | 43.7986 | 46.9153 | 44.7438 | 50.1227 | 50.9625 | 51.6285 | 55.2057 | 62.9204 |


From this table, it can clearly be seen that as the steps count increases, so 
does the average minutes to cook. The same relationship is present between 
ingredients count and average minutes to cook. So, it tracks that individually, 
both `n_steps` and `n_ingredients` have a positive association with minutes to 
cook, but variability in the other made this relationship difficult to see in 
the scatter plots. 

Nonetheless, aggregating by both shows a clear trend that indicates that yes, 
there is indeed a relationship between `n_steps`, `n_ingredients`, and minutes 
to cook a recipe. As `n_steps` and/or `n_ingredients` increases, so will the 
expected time to cook in minutes. 

To further illustrate this point, let’s see the earlier scatter plots with 
**AVERAGE** minutes to cook instead of raw minutes.

<iframe
  src="assets/avg_mins_steps_scatter.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>


This plot above shows a seemingly positive linear relationship until about 45 
steps, at which point it becomes way more variable and even begins to show what 
could be a negative linear relationship. This, along with the following graph, 
will be important moving forward.

The other graph with average minutes instead of raw minutes will be this 
scatter plot below, of `n_ingredients` (x) vs. average minutes to cook (y). 
Once again, we see a clearly positive linear relationship up until a certain 
point (ingredients = about 27 here), at which point a negative trend results.


<iframe
  src="assets/avg_mins_ingredients_scatter.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>





## Assessment of Missingness

Before assessing whether the missingness of any column is MAR or MCAR, I would 
first like to remind you of the columns that contain missing values, as well as 
indicate that for this section, I will be using the dataframe `df`, that 
contains all recipes (including duplicates). There are only 3 columns (from the 
original dataset, excluding the `avg_rating` column I added) that contain 
missingness. These columns are: `name`, `description`, and `rating`. Of these 
columns, I do not believe any of them are NMAR.

For the `description` column, I thought initially it may be, however upon 
further analysis I believe there may be some dependency on the `name` and 
`steps` columns. If the name is simple or revealing enough, which can be the 
case in some instances, there may be no need for a description. Examples of 
this include “peanut butter nana smoothie” and “spicy beef and vegetable soup,” 
which may not require descriptions. Others, with less intuitive names, such as 
“wasatch mountain chili” and “ultimate screwdriver” may be specific enough that 
those looking for those recipes already know what they are, or there is a clear 
description one google away. Because of this, I believe that the `description` 
may be missing dependent on the VALUES of the `name` column, and when the name 
is already clear or specific, it may be the case that no description is needed. 
However, to be sure, I would need to quantify the specificity and clarity of 
the names in some way, which could be prone to human bias. For now, I will say 
this is NMAR.

It could also be argued that the missing `name` is NMAR, since it could be 
dependent on the value of that name (recall that only one recipe is missing the 
name). However, I am more inclined to say this is an outlier and a specific 
case, and as such I will label it as MCAR since there isn't a trend in 
missingness in the `name` column.

The `rating` column is the one I want to focus on, since it has the most 
missing values (and really the only missing value count significant enough to 
the dataset size to matter at around 15,000). I first checked the `avg_rating` 
and `review` columns and found nothing out of the ordinary/any trend between 
the missingness and certain values and sentiments in these columns. Things got 
interesting, however, when I saw the average number of ingredients for recipes 
without missing ratings was less than the average number of ingredients for 
recipes missing ratings.

Taking the absolute (non-directional) difference in means between the average 
ingredient counts of the recipes with missing ratings and non-missing ratings 
as my test statistic, I proceeded to perform a permutation test with the 
following hypotheses: Null: recipes that have missing ratings have the same 
average ingredient count as recipes that have ratings; Alternative: Recipes 
with missing ratings have different average ingredient counts. I went in with a 
chosen significance level of 0.05, and after shuffling the missing/not missing 
rating labels and performing the test with 10,000 repetitions, I arrived with a 
p-value of 0.0 and rejected the null hypothesis in favor of the alternative. 
There was never a difference even near as large as my observed statistic. 

As such, I will adopt the belief that the missingness of the `rating` column is 
MAR dependent on `n_ingredients` with relative certainty. 

Below is a histogram showing this relationship. The vertical red line represents 
where the observed statistic fell in the distribution of calculated test 
differences in means. Notice how improbable the observed statistic is.


<iframe
  src="assets/empirical_means_mar.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>



But, just to be sure, I re-ran the permutation test with the Kolmogorov-Smirnov 
test statistic instead, just in case the difference in distribution shapes was 
getting in the way (and considering direction was not an issue). Operating 
under similar conditions (0.05 significance and 10,000 repetitions), another 
permutation test was carried out. 

My null hypothesis here was that the two distributions, rating missing and not 
missing, would have the same distribution of ingredient counts, and my 
alternative hypothesis was that these two values would be different. Once 
again, I got a p-value of 0.0 and promptly rejected the null hypothesis. 

It is once again evident that the missingness of `rating` is MAR dependent on 
`n_ingredients`, which I will assume moving forward. Below is the distribution 
of the K-S test statistics, along with a red line to once again represent the 
observed statistic’s placement on this distribution:


<iframe
  src="assets/empirical_ks_mar.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>


However, this doesn’t mean that the missingness of the `ratings` column is MAR 
dependent on every other column. For example, another permutation test, this 
time using the `minutes` column with the absolute difference in average means 
as the test statistic between the rating missing and not missing groups, shows 
a different relationship. The absolute difference in means is appropriate 
because I am not concerned with a direction, only a difference. Additionally, 
`minutes` are numerical.

Using a significance level of 0.05 again, my null hypothesis was that the 
distribution of `minutes` would be the same for the rating missing and not 
missing distributions, while my alternative hypothesis would be that they are 
different. After performing 10,000 repetitions of this permutation test, I came 
out with a p-value of 0.1138, which is above our significance threshold. 
Consequently, we fail to reject the null hypothesis and move forward with the 
assumption that the missingness of `rating` is not MAR dependent on the number 
of minutes a recipe takes to cook. 

Below is a histogram showing this relationship, with the red line once again 
representing the observed statistic relative to the other test stats.

<iframe
  src="assets/empirical_means_mcar.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>





## Hypothesis Testing

For my hypothesis testing, I wanted to do something related to my later 
prediction problem, so I decided to further analyze the relationship between 
cooking time and number of steps in a recipe. I chose the following specifics 
for framing this problem:

- **Null Hypothesis**: Cooking time in minutes is not dependent on the number of 
  steps in the recipe.
- **Alternative Hypothesis**: Recipes with higher step counts will take longer 
  to cook.
- **Test Statistic**: Directional difference in means (not_lower_half - 
  is_lower_half).
- **Significance Level**: 5%.

I think my alternative hypothesis is a reasonable one to test out because based 
on the pivot table at the end of my EDA, I saw this relationship when binning 
was applied. Now I want to see if it holds when there is no binning, and when 
`n_ingredients` are not grouped either. The directional difference in means was 
chosen as the test statistic because I need direction for this test, as my 
alternative hypothesis is meant to detect a difference if my belief that 
`n_steps` is positively associated with `minutes` holds. The order of 
subtraction was chosen for this reason as well. I chose 5% as my significance 
level partially because of convention, but also because it seems like a 
reasonable amount for a one-tailed test.

One other important characteristic of this permutation test I designed is how I 
set it up. Note that there are recipes that exist with step counts in all values 
from at least 1 to 38. After that (until the maximum step count recipes, 88), 
some counts have no recipes, but there are still 76 possible step counts in our 
dataset. As such, I determined the midpoint to be 38, grouped by the number of 
steps, assigned a new column `is_lower_half` to be a column of booleans 
representing whether the step counts are 38 or lower or not, and would proceed 
to shuffle this column for permutations. I did this to define what could be 
considered a “low” step count and a “high” step count for the purpose of 
categorization.

With 10,000 repetitions once again, I end up with a p-value of 0.0. With this 
value as low as it is, I will reject the null hypothesis in favor of the 
alternative hypothesis. It seems like recipes that have higher step counts will 
take longer to cook on average. This tracks with my assumptions.

Below is a histogram showing the distribution of the collected test statistics, 
with the vertical red line showing where the observed statistic lies.

<iframe
  src="assets/hypothesis_test.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>


## Framing a Prediction Problem

The prediction problem I will try to address is predicting the number of 
minutes it will take to complete a recipe. As `minutes` is numeric, and I have 
seen semi-linear relationships with the `minutes` column in my EDA, I will use 
regression to address this problem. 

My response variable is the number of minutes to cook, representing the time it 
takes to complete a recipe. For this, I will have filtered for recipes 
`<=300 minutes` to capture ideally only standing recipes that can be completed 
in one go and ignore certain outliers like the recipe that takes 1 million 
minutes.

I will use root mean squared error (RMSE) to evaluate the performance of my 
model due to it being interpretable and common, and because it is effective 
with regression models with quantitative predictions. I chose this over the 
R² score because RMSE is more common in model evaluation, and I want my results 
to be interpretable and directly comparable. I can also easily convert between 
the two.

The only columns we wouldn’t know at the time of prediction are `ratings`, 
`avg_rating`, `date`, and `user_id`. All other columns were present in 
`recipes`, the first dataframe, and were posted at the same time by the creator. 
I make sure to use none of these in my prediction algorithm.


## Baseline Model

My model is a regression model intended to predict the number of minutes it 
will take to complete a recipe. I currently only use two features, and base 
them both off of the pivot table at the end of my EDA. They are based on 
`n_steps` and `n_ingredients`, but they are each transformed with a 
`FunctionTransformer` for prediction to bin the counts, similar to the pivot 
table. 

I chose these features because they were the initial features I thought would 
best correlate with `minutes`, and I chose to bin them because of the earlier 
relationship I saw. After sending both columns into a `ColumnTransformer`, and 
then sending that into a pipeline followed by a `LinearRegression` model, I 
completed my baseline. Using a train-test split with a test size of 0.25 (to 
follow convention while trying to generalize as far as possible), I made my `X` 
and `y` which I would use moving forward in the project. Note that X is a subset
of my dataframe of the predictor columns, and y is the column I am trying to 
predict.

When evaluating my model with the RMSE, I got the value 43.897, which to me 
seems very high. I don’t think there are enough good features in this model to 
classify it as “good,” but it isn’t useless; there is a relationship that it is 
capturing, and there is a lot of variation in cooking time, which makes the
relationship difficult to capture.


## Final Model

So, the initial performance wasn’t the worst, but could definitely be improved. 
The first step was to edit my existing features and add some new ones.

First, the existing features lost a lot of information by binning. Because the 
`minutes` to cook also varied highly, it makes sense to also capture all of the 
`n_steps` and `n_ingredients`, so I opted to include these in my 
`ColumnTransformer` with passthrough. Since they are quantitative, they can be 
interpreted as is, but I also chose to apply a `StandardScaler` to make the 
found coefficients more comparable should the need arise.

Next, we will continue looking at those same columns, `n_steps` and 
`n_ingredients`. While I will still include all of the values, I want to be sure 
to help my model realize the shift in trends both columns show when they are 
plotted against average `minutes`. Recall my final scatter plots from EDA, 
showing linear trends and then completely changing at certain thresholds. An 
eye test reveals these thresholds to be 51 for `n_ingredients` and 28 for 
`n_steps`, though I may be a little off (not to worry; these are hyperparameters 
and can be tuned later). These thresholds work perfectly with `Binarizers`, the 
truth value of which should indicate the change in trends.

A new feature I added was the `calories` feature. I figured that more calories 
could indicate a larger meal, which could indicate more cooking time. I got the 
`calories` value by transforming the `nutrition` column and extracting the first 
element (`calories`), and then standardizing it with `StandardScaler` for 
comparability.

Continuing with the `nutrition` column, I found it the case that recipes that 
are higher in protein supposedly cook faster, and recipes with more fat 
supposedly cook slower. Conveniently, I have both of these values in the 
`nutrition` list, and after extracting them both, I came up with a metric called 
the protein-fat score, which is the difference between the two. I figure that 
this may be able to capture the impacts of time of both nutritional categories.

Next, I thought about how I may be able to use some text to discern 
information, and came up with two features. The first of which is called 
`vegetarian`, and to get it I added a column of booleans representing if a 
recipe was vegetarian or not. I got these booleans by checking if the `name` 
column or `tags` column contained any keywords that may indicate if a recipe is 
vegetarian or vegan. As those recipes tend to have more ingredients and less 
protein, I thought they would impact the cooking time due to my assumptions 
about those two categories. A similar process was done for the `quick` feature, 
where I looked in the `name` and `tags` columns for any words that may indicate 
convenience and speed. For each of these features, I used `OneHotEncoder`, and 
dropped one of each of the columns to avoid repetitive features.

Features aren’t everything my model needed, however. I also needed to tune 
hyperparameters, and the important ones I chose to fit were an intercept and 
the `Binarizer` thresholds for `n_steps` and `n_ingredients`. I used 
`GridSearchCV` to test values to see whether to fit an intercept or not, and to 
test values around my observed thresholds. I used 5-fold cross validation for 
generalizability and thoroughness, and even got to use negative root mean 
squared error as my scoring metric, which was perfect for my model evaluation. 
The optimal hyperparameters turned out to be an ingredients threshold of 28, a 
steps threshold of 60, and `True` for fitting an intercept. After assigning 
these optimal hyperparameters to variables and sending them into my final 
`ColumnTransformer` and `Pipeline`, I had my final model completed.

I used the same `X` and `y` I used on the baseline model for comparison and 
evaluation purposes, and fortunately my final model had improved on my baseline 
model. In the multiple times I ran the code, there was a difference in favor of 
the final model by anywhere between about 1.2 and 0.5 units between the two root 
mean squared errors (both generally hovered around the same 42.0 to 43.9 range). 
This may not seem like a lot, but keep in mind the variability of the `minutes` 
column, and how little error improves once a certain fit is met. Not only was 
this model able to improve training and test error, but it does so consistently 
for a variable that is hard to accurately predict due to several different 
competing measures, as evidenced by the original scatter plots. Therefore, this 
model is an improvement and can predict the time it takes to complete a recipe 
within about the time it takes to watch an episode of your favorite show.

## Fairness Analysis

I thought the most interesting groups to perform fairness analysis on would be 
those recipes that fall in the linear trend sections of the two scatter plots 
near the end of my EDA, and those that don’t. Specifically, my groups will be 
those recipes that have `n_ingredients` < 28 and `n_steps` < 55, and those that 
don’t meet at least one of those criteria.

Using a permutation test, I hoped to analyze this fairness. My null hypothesis 
is that the test RMSE is the same for recipes that meet the conditions defined 
above and those that don’t, while my alternative hypothesis is that the RMSE is 
less for those recipes that meet the conditions. I think this alternative 
hypothesis is most appropriate because the linear relationships were the best 
defined. Because this is directional, I will choose a difference in RMSE, with 
the order (doesn’t meet conditions - meets conditions). I think this is 
appropriate because it evaluates model performance on both groups. I also once 
again choose a significance level of 5% for conventional reasons, and will 
collect my test statistics and observed statistics on the test sets, while only 
training on the selected training data.

I repeated this permutation test for 1000 iterations (less because the test 
is very slow, but still encompassing), and got a p-value of 0.001. As such, I 
will reject the null hypothesis in favor of the alternative hypothesis. It seems 
to be the case that my model performs better on those recipes whose `n_steps` 
and `n_ingredients` counts are before the linear relationship threshold, which 
tracks with the graphs and expectations from earlier.

Below is a histogram showing the distribution of these differences. The
vertical red line represents where my observed statistic lies in this 
distribution. The distribution is quite normal, and yet our statistic
is still on the outer right tail

<iframe
  src="assets/fairness.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>