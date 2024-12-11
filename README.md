# Investigating-the-Relationship-between-Minutes-to-Cook-and-Number-of-Steps-in-a-Recipe
## DSC 80 Final Project
### By Daniel Mansperger


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
