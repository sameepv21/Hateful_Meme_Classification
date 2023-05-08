import React from 'react'

const Home = () => {
  return (
    <div className="flex flex-col items-center justify-center mt-12">
      <div className="max-w-md">
        <img className="mx-auto" src="Hateful_Meme.png" alt="Hateful Meme" />
      </div>
      <p className="text-left mt-4 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        Multimodal hateful meme classification is a complex and arduous task that requires the identification and
        classification of memes containing hate speech or offensive content by leveraging multiple modalities, such as
        text, images, and videos. The multifaceted nature of this task is primarily due to the fact that hateful memes
        can take different forms and combinations of modalities. In order to tackle this challenge, deep learning models
        can be employed to learn representations from multiple modalities and combine them to make accurate predictions.
        Furthermore, pre-trained models such as BERT, RoBERTa, or GPT-3 can be fine-tuned on a dataset of hateful memes
        to enhance their performance. The application of multimodal hateful meme classification is extensive and includes
        the integration of third-party APIs for social media platforms such as Twitter, Instagram, Stack Overflow, Reddit,
        and others.
      </p>
      {/* <p className="text-left mt-4 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        The emergence of hate speech and offensive content on social media platforms has become a major concern for 
        society. Hateful memes, in particular, have gained popularity in recent years due to their ability to spread 
        hate speech and misinformation in a humorous way. As a result, there is a growing need for automated tools to 
        identify and remove such content to ensure a safe and respectful online environment.
      </p> */}

      <p className='mb-4 text-left mt-4 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8'>
        The use of multimodal fusion techniques for hateful meme classification has shown promising results in previous
        research. By leveraging multiple modalities such as texts and images, these techniques can capture the nuances of
        hateful memes that may not be evident in a single modality alone. Facebook's hateful meme dataset provides a
        valuable resource for researchers to test and evaluate the effectiveness of these techniques in a real-world
        scenario.
      </p>
    </div>
  )
}

export default Home