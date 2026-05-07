# ML image classificator for material handling machine

## Overview

Basic ide a to implement pipeline of sampling image from conveyor, preprocess it and push to model to get in return 1 of 3 possible states:
- Empty
- Pallet
- Pallet and Block

## Model development

### Image dataset

Before start i collected and separeted image dataset for this 3 classes. This dataset quite small: 161 Empty, 159 Pallet, 203 Pallet and Block images.
Images were taken on same equipment with variable light setups by highliting with flashlights or by adding extra shadows. Additionally pallet and blocks were moved randomly in area to get more variable images. 

<!-- Inset raw image -->

### image preprocessing 

Initial idea was to crop by 1/3 from each side because required objects usualy located in central vertical part of image.
However, my data set is to small for learning from the scratch and equipment setup and positions block possibility to expand it by mirror flipping and big angles rotation. by those reasons I decided to use pretrained model and image should match size and color channel. In my case it is 224x224 pixels and RGB. 
Anyway images in dataset in FullHD resolution so for I resize it and crop by 
