# Pokemon-Type-Classifier
A repository containing resources, processes and findings for my Pokemon Type Classifier.

## Motivation
With the rise of Pokémon GO as a mobile game, many people that never owned old game consoles to play classic Pokémon games are getting exposed to the franchise. However, the emphasis on the 'catching' aspect of Pokémon makes people less concerned with the battle mechanics that were rampant in older editions of the game. The most elementary of these was type advantages, essentially using 'more effective' types against others in order to deal more damage or receive less damage. 

Our goal with this project is to determine if we can use a convolutional neural network to determine a Pokémon's type from any image of a Pokémon in an attempt to possibly alert players when the opposing Pokémon poses a threat to the user's Pokémon. Once we train a neural network to recognize types from images, we can then hard code the type advantages as conditionals to alert the player.

## The Data
The data is hosted on pokemondb.net/pokedex/national. We scraped 8360 images of each Pokémon with their corresponding type and name in order to create labels. We created a dataframe that could be referenced when images are being loaded, and saved each image in a folder corresponding to their types. Most of the images are 128x128 and in a .png format. Certain images that are 'official artwork' are of greater dimensions and in a .jpg format. 

## Exploratory Data Analysis
### 
