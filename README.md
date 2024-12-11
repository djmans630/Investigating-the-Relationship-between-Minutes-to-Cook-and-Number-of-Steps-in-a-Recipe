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
