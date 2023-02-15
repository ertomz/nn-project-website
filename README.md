# Designing Tattoos with a Neural Network

## Related Works

### How We Built a Neural Network for Tattoo Style Recognition

Vuksic, Goran. “Ai and Tattoos: How We Built a Neural Network for Tattoo Style Recognition.”, _Medium_, 25 Aug. 2017, https://blog.tattoodo.io/ai-and-tattoos-how-we-built-a-neural-network-for-tattoo-style-recognition-6e641df99a05. 
Vuksic’s 2017 blog post describes the process of developing Tattoodo, an online platform for tattoo art that now utilized a neural network to recognize and classify different tattoo styles. They offer users a personalized tattoo feed based on preferred artists, styles (such as tribal, water color, or traditional), and motifs (such as flora, swords, or dragons). Developers used a deep learning network called Caffe along with Nvidia’s Deep Learning GPU Training System to train the neural network.

### Man Gets Tattoo of Art Created by Neural Network

Hood, Lonnie Lee. “Man Gets Tattoo of Art Created by Neural Network”, _Futurism_, 10 Apr. 2022, https://futurism.com/the-byte/tattoo-created-by-neural-network. 
Hood’s recent 2022 article talks about DALL-E2, a neural network created by OpenAI that “turns natural human language into realistic photos and art,” (Hood, 2022). One man decided to ask the network to design him a simple tattoo of the letter A which he then got tattooed on his arm. Due to potential network biases, some worry that DALL-E2 could end up being “some kind of branding machine,” especially if people continue to get tattoos of images it generates.

### Artificially Generated Tattoo

Betin, Vasily. “Artificially Generated Tattoo”, _Medium_, 12 Mar. 2020, https://medium.com/vasily-betin/artificially-generated-tattoo-2d5fbe0f5146. 
In his 2020 Medium post Vasily Betin describes how his love for tattoos and technology led him to create his own tattoo generation model. He outlines how he created his dataset of images of tattoos by scraping Instagram and Pinterest, cleaned the data, trained the network on low then high resolution images, and then continued to upscale and perfect the model. After generating over 2,000 images, Vasily chose his favorites and actually got one tattooed on his arm. The model is released on RunwayML with a public Github Repo. 

### Sketch-a-Net: A Deep Neural Network that Beats Humans

Yu et. al. “Sketch-a-Net: A Deep Neural Network that Beats Humans”, _Springer Science & Business Media New York 2016_, 26 July 2016, https://link.springer.com/content/pdf/10.1007/s11263-016-0932-3.pdf
Many tattoos are actually sketches - having edges but lacking visual cues such as color and texture.  The paper presents Sketch-a-Net, which is designed for sketch rather than natural photo statistics, exploits the unique sketch-domain properties to modify and synthesize sketch training data and therefore increase the volume and diversity of sketches for training, and explores different network ensemble fusion strategies.

### Generative Art: Between the Nodes of Neuron Networks

Caldas Vianna, Bruno. “Generative Art: Between the Nodes of Neuron Networks.” A_rtnodes, no. 26_, 2020, https://doi.org/10.7238/a.v0i26.3350. 
Caldas Vianna’s 2002 paper on generative art discusses the history of GANs (generative adversarial networks such as StyleGAN2) that can generate images. Without clear answers, they pose questions about the originality of generated art and whether or not we can formalize aesthetics by putting certain rules into a model. They question whose art it is, as it may be unique but not totally original. Finally, Caldas Vianna seems to think that right now images created by NNs will not be mistaken for replicas or real art – they’re not there yet, but that’s a question for the future.

### Can: Creative adversarial networks, generating “art” by learning about styles and deviating from style norms

Elgammal, Ahmed, et al. "Can: Creative adversarial networks, generating “art” by learning about styles and deviating from style norms." 2017. https://arxiv.org/abs/1706.07068
Elgammal et al.’s paper on using General Adversarial Networks (GAN) to produce art which is as unique to other artists’ works as possible. Their research found that they could generate art from other designs and drawings which were indistinguishable from works found in art galleries to human subjects, and even sometimes preferred to human works. 

### Tattoo-GAN

This one is on github so it has some open sources: 
https://github.com/silkdom/Tattoo-GAN
This project uses generative adversarial network models  (DCGAN & StyleGAN2) to generate tattoo images. It created a tattoo dataset by first compiling a potential list of tattoo artists on instagram and scraping the contents they produced. It adapts a state of the art architecture, Nvidia's StyleGAN2 (Repo), and trains the dataset on a VM (500GB / P5000 GPU). 

### State of the arts (online AI tattoo generators)
https://www.tattoosai.com/explore
https://neural.love/ai-art-generator/1ed20058-8956-6ef0-86b4-f9527de7d626/tattoo-made-by-ai
https://aitattoo.net
https://www.tattoojenny.com

###

## Introduction

### Introductory Paragraph 

Current art generators can produce tattoo recommendations which can be considered unrealistic and intimidating, with emphasis on artistic values over reasonable tattoo recommendations. Additionally, many people get similar tattoos that are popular or trending. We aim to create a model trained solely on images of tattoos that can produce unique tattoo designs based on a user’s preferences.

### Background Paragraph 

This is an interesting problem to address simply because it has barely been done. There are many models that are trained off artwork and can create something original, but only rarely has a model like this been trained solely on images of tattoos. We only found two people that have attempted this, but their models have not been used widely if at all beyond for personal purposes. Beyond this, tattoo generation is a difficult problem to address because it poses many ethical issues, as outlined in the ethics section later on.

### Transition Paragraph

To make the generated tattoo images more realistic, we can augment traditional image generation neural networks with parameters like color, simplicity of the style, location, and size.

### Details Paragraph

Possible technical challenges of our project include a) finding explicit databases of images of tattoos b) implementation of image scraping and cleaning c) exploring and improving available training model on tattoo images. 

### Assessment Paragraph

This project (hopefully) successfully built a tattoo generation model which was capable of producing tattoos based upon user input while producing uniquely designed art which attempted to avoid plagiarism of other artistic works. 


## Ethics
Looking at the ethical issues that might be raised from this project there are a few things that we can think about. First, however, is looking at whether we should be doing this to begin with. Since it is a class project that is interesting to us, we believe that we should be doing this. Nevertheless, in real life, there should be a deeper thought and restraints/precautions for this project–acknowledgement of the datasets, sources, and artists, as well as age restriction, and the words that can be used for our Tattoo generator. Additionally, we need to consider the diversity of our team. We are not representative of all skin colors, we come from cultures and families that don’t support tattoos, and none of us have tattoos. 

There are a few ethical issues with our project which are outlaid here:

**Copyrights→** As we are dealing with artistic pieces, which are generated by Tattoo artists who usually have copyrights over those pieces of art.We need to make sure that the data we use in our dataset does not conflict with copyrights. It is also important to ask how Tattoo artists may feel with such a program, which is taking away part of their living and artistic value.

**Skin color for Data→** Our data needs to be diverse enough for it to include different types of skin color. Black vs white would not be sufficient here, and the data does need to aim for a more diverse set. We would say that a guideline of about 100 shades of skin color is the minimum (though we don’t promise to deliver that in this project).

**Encouraging young people to get a tattoo→** Such a program may encourage young people to have a tattoo at an earlier age. Just having such a program may give this incentive, but also the reduced cost due to having a final tattoo proposal instead of having the tattoo artists draw one.

**Indigenous culture→** Some cultures have a higher meaning of having tattoos than just body paintings. In these cultures, having a tattoo can have a meaning of maturity, tribal connection, stage of life, age, genealogical connection and many others. For this project, we would say that it is ok to use such data in our dataset. We would put a disclaimer regarding such an issue, and will try to acknowledge tribal connection and information when adequate.

**Offensive symbols→** By allowing users to specify what type of tattoo they want generated, there is potential for the creation (accidental or not) of offensive words and symbols. The dataset the model is trained on may contain tattoos with offensive symbols, but for the scope of this project we do not have the capacity to screen our dataset as thoroughly as we would like.

**What words are used→** Since many existing tattoos may contain words of profanity, the data we trained on may lead to tattoo generations with hatred and disrespect. We need to ensure that generated tattoos avoid discriminating words or terms which are offensive to groups. 

**Providing the same tattoo over and over again→** This may be a more personal issue for private users who want to have a distinct tattoo as much as possible. We cannot guarantee that the program will not supply the exact same tattoo, which should also be added a a disclaimer, as well as providing the program with some variation parameter, that will make slight changes between every graphic we provide.


Disclaimer: Our model attempts to avoid plagiarizing from other artists’ works and attempts to avoid generating offensive art. We advise that users… :)


## Problem

Tattoos are personal, permanent, and pretty hard to design yourself. While it's exciting to engage in the intimate process of working with a tattoo artist to design your very own tattoo, that process can take a long time and cost a lot of money. Worth it? Perhaps. But what if you have an idea for a tattoo but you just don't quite know where to go with it? That's where Neural Nets might come into play...

This has interesting potential because of the recent generative art AI models. Sites like Midjourney or HotpotAI generate images from text that a user types in. They are artificially generated and realitively authentic pieces of art produced from models trained on thousands of images of art. But, there is some controversy around these models as artists grow frustrated with these models mimiking their work. It raises questions about the ethics of producing these images and about what real artwork actually is.


## My Idea

I propose a neural net that is trained on a dataset of a wide range of images of tattoos.

Much of the existing work around neural networks and tattoos revolve around classification. Tattoodo is a platform that allows its 1 million users to search for tattos based on styles, motifs, and artists they like. Employees trained a neural network to recognize and classify different tattoos based on styles and motifs such as water color, traditional, dragons, flora, etc. They used Caffe, a deep learning framework, and Nvidia's Deep Learning GPU Training System [[3]](#3). However, this is just classification, not generation. 

While art generating models exist, there are no major ones that are specifically trained on images of tattoos. You can ask Midjourney, for example, for images of a "simple flower tattoo on a woman's arm", as seen below, but no major model is trained solely on tattoos. 

![Midjourney 1](/images/midjourney1.png)
![Midjourney 2](/images/midjourney2.png)

There are some smaller models, however, that exist but have very rarely been used.

In 2020 Vasily Betin created StyleGan2, which does generate unique tattoos from his own scraped and cleaned Pinterest image dataset, but it is not widely used and seems like it could be improved. However, he ended actually getting a tattoo that his model generated [[1]](#1). The main article I found about it was a Medium post by the creator. 

![StyleGan2](/images/StyleGan2.png)

In 2022 a man named Everett Randle used Dall-E2, a model created by OpenAI that creates photos and art from human language, to generate a tattoo. Not only that, but he actually then got that tattoo [[2]](#2). The article claims that this was the first time someone got a real tattoo that a model produced, although it appears as though Betin beat him to it. 

![Dall-E2](/images/Dall-E2.png)

Thus, there appears to be a lot of potential to produce someting new and better and specifically focus it on tattoos.


## Ethics

This problem is even more interesting because of the ethics around what images are used to train the model. 

Skin color comes to mind first – how many images in the dataset are of tattoos on dark skin versus light skin?

Another consideration is training the model on tattoos that have indigenous roots. It may be problematic for the model to create new tattoos that mimic tribal and other indigenous tattoos.

When Everett Randle got the tattoo created by DALL-E2, people started to worry that the model could become "some kind of branding machine" [[2]](#2). I agree, but I also think that this could be addressed by managing what images are used in training and offering complete transparensy with how the model is trained.


## Database

I could not find any explicit databases of images of tattoos (one might work but it is the preliminary database for identifying criminals based on their tattoos). In his 2020 Medium post, Betin walks through how he designed StyleGan2, including how they collected the tattoo images and then scraped and cleaned the data [[1]](#1). This process could be mimicked and likely improved in this project.


## Project Goals

1. Create (if can't find) a large dataset of images of tattoos. 
2. Learn about image scraping and cleaning.
3. Research and potentailly improve upon pre-existing art generation models and train it specifically on the tattoo dataset.
4. Research various opinions around the ethics of producing arificial art.
5. Have fun!
6. Get a tattoo! 


## References
<a id="1">[1]</a>
Betin, Vasily. “Artificially Generated Tattoo.” Artificially Generated Tattoo, Medium, 12 Mar. 2020, https://medium.com/vasily-betin/artificially-generated-tattoo-2d5fbe0f5146. 

<a id="2">[2]</a>
Hood, Lonnie Lee. “Man Gets Tattoo of Art Created by Neural Network.” Man Gets Tattoo of Art Created By Neural Network, Futurism, 10 Apr. 2022, https://futurism.com/the-byte/tattoo-created-by-neural-network. 

<a id="3">[3]</a>
Schweitzer, Annette. “How Ai Can Help You Find the Perfect Tattoo.” Getting Good Ink: How AI Can Help You Find the Perfect Tattoo, NVIDIA Blog, 29 Sept. 2017, https://blogs.nvidia.com/blog/2017/09/29/find-the-perfect-tattoo/. 

<a id="4">[4]</a>
Vuksic, Goran. “Ai and Tattoos: How We Built a Neural Network for Tattoo Style Recognition.” AI and Tattoos: How We Built a Neural Network for Tattoo Style Recognition, Medium, 25 Aug. 2017, https://blog.tattoodo.io/ai-and-tattoos-how-we-built-a-neural-network-for-tattoo-style-recognition-6e641df99a05. 
