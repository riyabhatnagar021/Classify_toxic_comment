# <b>Classify_toxic_comment</b>


We built a multi-headed model that’s capable of detecting different types of of toxicity like
threats, obscenity, insults, and identity-based hate better than Perspective’s current models. 

<b> DATA </b>

We hqave a large number of Wikipedia Comments which have been labeled by human raters for toxic behaviour. The types of toxicity are
:
* Toxic

* Severe_toxic

* Obscene

* Threat

* Insult

* Identity_hate

You must create a model which predicts a probablity of each type of toxicity for each comment.

<b> Result </b>


LSTM is used to build the above model. 


Test Accuracy A cieved: 95.9%


# <b>Deployment using Flask</b>

Project Structure
This project has Three major parts :

1.Comment_toxicity_prediction.ipynb - This contains code for our model to predict the toxicity of user's comment.


2.apptoxic.py - This contains Flask APIs that receives the user's comments, computes the toxicity based on our model and returns it.


3.templates - This folder contains the HTML template to allow user to enter the comment and displays the toxicity of that comment.

<b>Running the Project</b>

Run apptoxic.py 

You should be able to view the homepage as below :
![six](https://user-images.githubusercontent.com/50323219/66721508-a8830100-ee21-11e9-98e3-ab029e9427d4.JPG)

Enter Your Comment in the box and hit Predict


The Toxicity of your comment will be displayed:

![five](https://user-images.githubusercontent.com/50323219/66721600-eb44d900-ee21-11e9-8881-7f2280b432c9.JPG)



