# Overview of the internship
I have worked as a Solutions Architect intern in public sector for five months, from the 4<sup>th</sup> of April until the 26<sup>th</sup> of August. Please find here the work done during my internship at Amazon Web Services
## Project
### Goal & Challenges
This project consists of developing a voice synthetizer tool, like the Polly service available on the AWS console. 
This project had two main objectives. 
- The first one is to create a voice synthesizer that can be customised by AWS clients using their own dataset and AWS services. The AWS service Polly does not have a customizable voice, hence clients may need a solution to generate their own custom voice. 

- An additional goal was to be able to synthetize emotions by training a pre-trained model with a dataset containing emotions. To simplify the problem, the aim was to generate an angry voice synthetizer. The service that could result from this project would take as input a text, which would ba analyzed by an AWS service to determine the emotions present in the text. Thus the text would be synthetized using the trained model corresponding to the emotion. 

Before reaching these goals, there is a main challenge: using a git repository that was developed on another platform, on the AWS services.

### Steps
To tackle this challenge, I used a git repository to synthetize voice from text. First I ran the code using AWS Sagemaker Jupyter Notebook and I used pretrained models to synthetize text. Then I launched my own training. I attempted to optimize the training by following SageMaker's good practices. To do so, I have used the tools developed for SageMaker: a pytorch estimator. As the code I was using required a custom environment, I had to extend a prebuilt container. However because of the size of the project, this task required more attention and the container was extended using an EC2 instance. 

### Deep dive in the project

To know more about the work done for this project, please see the following files: 

- **Overview_Internship.md - Current file**: Overview and general information of the internship
    1. [**Step1_DevelopmentProject.md**](./files/Step1_DevelopmentProject.md): Gives the details of the development of the project.
    2. [**Step2_ScaleProject.md**](./files/Step2_ScaleProject.md): Gives the details on how I intend to scale the project.
    3. [**TTS_Explanations.md**](./files/TTS_Explanations.md): Gives information on how Text-to-speech models work.


## AWS Achievements
- [x] Awesome Builder 1 (AB1) (Mandatory)
- [ ] Awesome Builder 2 (AB2) (Optional)
- [ ] Awesome Builder 3 (AB3) (Optional)
- [x] Cloud Practicioner Certification (Mandatory)
- [x] Solution Architect Associate Certification (Optional)
- [ ] Machine Learning Specialty Certification (Optional) - Work in progress

## Social events
Throughout my internship I had the apportunity to attend several event with Amazon Web Services. 

### AWS Paris Summit 2022
On the 12<sup>th</sup> of April, AWS organised the AWS Paris Summit, regroupings many teams from AWS, partners anc clients. At this event, I realised the global impact that AWS has and the size of the community that gravitates around the company. The projects made possible thanks to AWS services are numerous and very impressive. I attended a presentation by Teads, one of AWS's client, which is working with AWS on improving the transparency and [estimation of the carbon footprint](https://engineering.teads.com/sustainability/carbon-footprint-estimator-for-aws-instances/
) resulting from using the AWS cloud. They are working on improving the AWS [Customer Carbon Footprint Tool](https://aws.amazon.com/fr/blogs/aws/new-customer-carbon-footprint-tool/) available in the AWS console.  

### World Wide Public Sector Tech Builder Conference
During my internship I had the great opportunity to attend the first World Wide Public Sector Tech Conference for women and non-binary people in Arlington from the 25<sup>th</sup> to the 27<sup>th</sup> of May. This was a wonderful opportunity for me to meet incredible Amazonians, learn more about the work of a Solutions Architect and about Amazon Web Services. I highly recommend this conference. I have written a [*Trip report*](./files/Washington_TripReport.pdf) following the conference for more details on this.

### Forum Teratec 2022
I have attended the Teratec Forum at Ecole Polytechnique on the 14<sup>th</sup> of June. This forum brings together international experts in High Performance Computing, Simulation, Big Data and Artificial Intelligence. I got the opportunity to listen to Mathieu Jeandron, Technical Lead in Public Sector and Senior SA Manager,  [answer questions](https://teratec.eu/forum/Table_Ronde_02.html) about data sovereignty and security in the cloud and how AWS provides High Perfomance Computing and Artificial Intelligence services to meet its clients' needs.

### VivaTech
The VivaTech forum took place from the 15<sup>th</sup> until the 18<sup>th</sup> of June. I got the opportunity to exchange with student interested in the Solutions Architect positions. I also shared with them the recruitment process for internships.

### Amazon's LinkedIn post for internships at AWS
I gave my testimony on my internship as a Solutions Architect to promote internships at AWS. This testimony was shared on Amazon's [Linkedin post](https://www.linkedin.com/posts/amazon_title-activity-6959810001741746176-WjPS/).

## Acknowledgments

I would like to express my sincere gratitude to my supervisors Guillaume Neau and Mathieu Jeandron for giving me the opportunity to work on such a challenging and exciting project and for encouraging me to participate in Amazon and AWS social events.
I am grateful to Guillaume and my mentor Olivier Sutter for guiding me through this project and helping me overcome the hardships I encountered, while training me as a Solutions Architect.