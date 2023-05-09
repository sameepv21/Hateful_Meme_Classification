import React from 'react'
import { Link } from "react-router-dom";

const Home = () => {
  return (
    <div className="flex flex-col items-center justify-center mt-4">
      <div className="flex justify-center mb-4">
        <h1 className="text-4xl font-bold text-center text-gray-800">
          What is it?
        </h1>
      </div>
      <div className="max-w-md">
        <img className="mx-auto" src="Hateful_Meme.png" alt="Hateful Meme" />
      </div>
      <p className="text-left mt-4 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        Multimodal hateful meme classification is a complex and arduous task that requires the identification and
        classification of memes containing hate speech or offensive content by leveraging multiple modalities, such as
        text, images, and videos. The multifaceted nature of this task is primarily due to the fact that hateful memes
        can take different forms and combinations of modalities. <span className='font-bold'>In order to tackle this challenge, deep learning models
          can be employed to learn representations from multiple modalities and combine them to make accurate predictions</span>.
        Furthermore, pre-trained models such as <span className="font-bold">BERT, RoBERTa, or GPT-3</span> can be fine-tuned on a dataset of hateful memes
        to enhance their performance. The application of multimodal hateful meme classification is extensive and includes
        the integration of third-party APIs for social media platforms such as Twitter, Instagram, Stack Overflow, Reddit,
        and others. 
        <Link
          to="/project"
          className="text-blue-500 hover:text-blue-600 font-bold ml-1"
        >
          Try me
        </Link>
      </p>

      <div className="flex justify-center mt-4">
        <h1 className="text-4xl font-bold text-center text-gray-800">
          Related Work
        </h1>
      </div>

      <p className='text-left mt-4 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8'>
        The use of multimodal fusion techniques for hateful meme classification has shown promising results in previous
        research. By leveraging multiple modalities such as texts and images, these techniques can capture the nuances of
        hateful memes that may not be evident in a single modality alone. Facebook's hateful meme dataset provides a
        valuable resource for researchers to test and evaluate the effectiveness of these techniques in a real-world
        scenario.

        State-of-the-art model, which is the winner of facebook hateful meme classification, uses an ensemble <span className = 'font-bold'>
        UNITER, ViT etc.</span> and is able to achieve <span className='font-bold'>74%</span> on the test set which is 
        far <span className = 'font-bold'>lesser than human performance of 84%.</span>
      </p>
    </div>
  )
}

export default Home