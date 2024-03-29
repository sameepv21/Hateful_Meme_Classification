import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [image, setImage] = useState(null);
  const [prediction, setPrediction] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = (event) => {
    event.preventDefault();
    setIsLoading(true);
    const formData = new FormData();
    formData.append('file', image);
    axios.post('http://localhost:8000/predict', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    })
      .then(response => {
        console.log(response.data);
        if (response.data.prediction === 0) {
          setPrediction('Not Hateful');
        } else {
          setPrediction('Hateful');
        }
        setIsLoading(false);
      })
      .catch(error => {
        console.log(error);
        setIsLoading(false);
      });
  };

  const handleImageChange = (event) => {
    setImage(event.target.files[0]);
  };

  return (
    <div className='bg-gray-100'>
      <div className='flex justify-center mt-5 text-4xl font-bold text-violet-800'>
        Bring your vision to life - upload and test your images now
      </div>
      <div className="min-h-screen flex flex-col justify-center items-center">
        
        <form className="bg-white shadow-md rounded px-8 pt-6 pb-8 mb-4" onSubmit={handleSubmit}>
          <div className="mb-4">
            <label className="block text-gray-700 font-bold mb-2" htmlFor="image">
              Image
            </label>
            <input className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline" id="image" type="file" onChange={handleImageChange} required />
          </div>
          <div className="flex items-center justify-between">
            <button className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline" type="submit">
              Submit
            </button>
          </div>
        </form>
        {isLoading && (
          <div className="bg-white shadow-md rounded px-8 pt-6 pb-8 mb-4">
            <h1 className="block text-gray-700 font-bold mb-2">Predicting...</h1>
          </div>
        )}
        {prediction !== '' && !isLoading && (
          <div className="bg-white shadow-md rounded px-8 pt-6 pb-8 mb-4">
            <h1 className="block text-gray-700 font-bold mb-2">Prediction</h1>
            <p className="text-gray-700">{prediction}</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
