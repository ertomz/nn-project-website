# Designing Tattoos with a Neural Network


## Problem

Tattoos are personal, permanent, and pretty hard to design yourself. While it's exciting to engage in the intimate process of working with a tattoo artist to design your very own tattoo, that process can take a long time and cost a lot of money. Worth it? Perhaps. But what if you have an idea for a tattoo but you just don't quite know where to go with it? That's where Neural Nets might come into play...

This has interesting potential because of the recent generative art AI models. Sites like Midjourney or HotpotAI generate images from text that a user types in. They are artificially generated and realitively authentic pieces of art produced from models trained on thousands of images of art. But, there is some controversy around these models as artists grow frustrated with these models mimiking their work. It raises questions about the ethics of producing these images and about what real artwork actually is.


## My Idea

I propose a neural net that is trained on a dataset of a wide range of images of tattoos.

Much of the existing work around neural networks and tattoos revolve around classification. Tattoodo is a platform that allows its 1 million users to search for tattos based on styles, motifs, and artists they like. Employees trained a neural network to recognize and classify different tattoos based on styles and motifs such as water color, traditional, dragons, flora, etc. They used Caffe, a deep learning framework, and Nvidia's Deep Learning GPU Training System (Vuksic, 2017). However, this is just classification, not generation. 

While art generating models exist, there are no major ones that are specifically trained on images of tattoos. You can ask Midjourney, for example, for images of a "simple flower tattoo on a woman's arm", as seen below, but no major model is trained solely on tattoos. 

![Midjourney image 1](/images/midjourney1.png){width=100px, height=200px}

![Midjourney image 2](/images/midjourney2.png)

There are some smaller models, however, that exist but have very rarely been used.

In 2020 Vasily Betin created StyleGan2, which does generate unique tattoos from his own scraped and cleaned Pinterest image dataset, but it is not widely used and seems like it could be improved. However, he ended actually getting a tattoo that his model generated. The main article I found about it was a Medium post by the creator. 

![StyleGan2](/images/StyleGan2.png)

In 2022 a man named Everett Randle used Dall-E2, a model created by OpenAI that creates photos and art from human language, to generate a tattoo. Not only that, but he actually then got that tattoo (Hood, 2022). The article claims that this was the first time someone got a real tattoo that a model produced, although it appears as though Betin beat him to it. 

![StyleGan2](/images/Dall-E2.png)


Thus, there appears to be a lot of potential to produce someting new and better and specifically focus it on tattoos.


## Ethics

This problem is even more interesting because of the ethics around what images are used to train the model. 

Skin color comes to mind first – how many images in the dataset are of tattoos on dark skin versus light skin?

Another consideration is training the model on tattoos that have indigenous roots. It may be problematic for the model to create new tattoos that mimic tribal and other indigenous tattoos.

When Everett Randle got the tattoo created by DALL-E2, people started to worry that the model could become "some kind of branding machine" (Hood, 2022). I agree, but I also think that this could be addressed by managing what images are used in training and offering complete transparensy with how the model is trained.


## Database

I could not find any explicit databases of images of tattoos (one might work but it is the preliminary database for identifying criminals based on their tattoos). In their 2020 Medium post, Betin walks through how they designed StyleGan2, including how they collected the tattoo images and then scraped and cleaned the data. This process could be mimicked and likely improved in this project.
