# Pokemon-Type-Classifier
A repository containing resources, processes and findings for my Pokemon Type Classifier.

## Motivation
With the rise of Pokémon GO as a mobile game, many people that never owned old game consoles to play classic Pokémon games are getting exposed to the franchise. However, the emphasis on the 'catching' aspect of Pokémon makes people less concerned with the battle mechanics that were rampant in older editions of the game. The most elementary of these was type advantages, essentially using 'more effective' types against others in order to deal more damage or receive less damage. 

Our goal with this project is to determine if we can use a convolutional neural network to determine a Pokémon's type from any image of a Pokémon in an attempt to possibly alert players when the opposing Pokémon poses a threat to the user's Pokémon. Once we train a neural network to recognize types from images, we can then hard code the type advantages as conditionals to alert the player.

## The Data
The data is hosted on pokemondb.net/pokedex/national. We scraped 8360 images of each Pokémon with their corresponding type and name in order to create labels. We created a dataframe that could be referenced when images are being loaded, and saved each image in a folder corresponding to their types. Most of the images are 128x128 and in a .png format. Certain images that are 'official artwork' are of greater dimensions and in a .jpg format. 

## Exploratory Data Analysis
### Data Cleaning
As these are images, we had very few roadbumps during this stage of theprocess. Utilizing ```Pillow``` (Python Imaging Library), we were able to scrape the images and remove the transparency layer from them before saving them into our folders. We chose an image size of 256x256 to feed into the neural network for two reasons: most of the images are 128x128 (square), the few that aren't square are around 256x256, and we want to save as much information as possible. 

### Relevant Metrics
We determine our most relevant metrics to be precision and recall. High precision will help determine 'bad switches', or falsely switching Pokémon when there is no threat, and High recall will help determine bad ‘stay-ins’ or failing to identify a threat that most likely results in your Pokémon being defeated. Although they provide free revives for your fainted Pokémon by playing daily, players that play more frequently will likely run out. It's important for us to maximize recall as this will most likely result in a fainted Pokémon and can possibly cost a player real-life currency.

### Data Processing
In order to access our model accuracy, we split our data set into 3 categories: train, validation and test. We load the images in ```X``` and cross-reference the image name with a dataframe that labels it in ```y```. We utilized a 20% train/test split and stratified the data. In addition, we created class weights using the number of images that are in the data set by category.

## Model Analysis
### Hyperparameter Tuning
Since my computer can't handle complex grid searches, we did a 'manual' grid search and determined our most effective hyperparameters to be:

Input Shape: (256,256,3)

Kernel Size: (3,3)

Pool Size: (2,2)

Dropout: 0.25 for Convolutional Layers and 0.5 for our Dense Layers

Activation Layer: ReLu

Final Activation Layer: Sigmoid

L2 Regularlizer: 0.01

Batch Size: 16

Class Weights: Calculated based on class distribution ratios


Optimizer: Adam

Loss: Binary Cross-Entropy

Metrics: (Accuracy), Precision, Recall (Threshold = 0.4)

### Baseline Model
This was conducted using a dummy classifier using a stratified strategy.
It's performance was:

Precision: 11.72%

Recall: 11.85%

### Model Structure
Our CNN is made up of 4 convolution layers with pooling and dropout after each layer, and 3 dense layers before classification. For multi-label classification, it is import to note that binary cross-entropy was vital in ensuring the model runs properly. 


### Model Performance
Our model took a significantly long time to train as it did not show signs of improvement until the 18th epoch, and took 200 epochs to achieve results I was satisfactory with. 

As we can see below, our model performed better than the baseline across both metrics; however, when compared to the training data performance, we can see that our model is most likely overfit to the training set. This may have something to do with the poor model performance on testing data. 
Test Precision: 83.90%
Test Recall: 48.63%
