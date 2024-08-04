'use client';

import React, { useState } from 'react';
import axios from 'axios';

const StockGraph = () => {
  const [stockName, setStockName] = useState('');
  const [forecastDays, setForecastDays] = useState('');
  const [modelType, setModelType] = useState('1');
  const [graphImage, setGraphImage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');
    try {
      const response = await axios.post('http://localhost:5000/predict', {
        stockName,
        forecastDays,
        modelType
      });
      setGraphImage(response.data.image);
    } catch (error) {
      console.error('Error fetching graph:', error);
      setError('Failed to generate graph. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="p-4 max-w-2xl mx-auto">
      <h2 className="text-2xl font-bold mb-4">Stock Prediction</h2>
      <form onSubmit={handleSubmit} className="mb-4 space-y-4">
        <div>
          <label htmlFor="stockName" className="block mb-1">Stock Name:</label>
          <input
            id="stockName"
            type="text"
            value={stockName}
            onChange={(e) => setStockName(e.target.value)}
            placeholder="e.g., AAPL"
            className="w-full p-2 border rounded"
            required
          />
        </div>
        <div>
          <label htmlFor="forecastDays" className="block mb-1">Days to Predict:</label>
          <input
            id="forecastDays"
            type="number"
            value={forecastDays}
            onChange={(e) => setForecastDays(e.target.value)}
            placeholder="e.g., 30"
            className="w-full p-2 border rounded"
            required
            min="1"
          />
        </div>
        <div>
          <label htmlFor="modelType" className="block mb-1">Prediction Model:</label>
          <select
            id="modelType"
            value={modelType}
            onChange={(e) => setModelType(e.target.value)}
            className="w-full p-2 border rounded"
          >
            <option value="1">Linear Regression</option>
            <option value="2">SVR</option>
            <option value="3">LSTM</option>
          </select>
        </div>
        <button 
          type="submit" 
          className="w-full p-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition"
          disabled={isLoading}
        >
          {isLoading ? 'Generating...' : 'Generate Graph'}
        </button>
      </form>
      {error && <p className="text-red-500 mb-4">{error}</p>}
      {graphImage && (
        <div>
          <h3 className="text-xl font-semibold mb-2">Prediction Graph</h3>
          <img 
            src={`data:image/png;base64,${graphImage}`} 
            alt="Stock Prediction Graph" 
            className="max-w-full h-auto border rounded"
          />
        </div>
      )}
    </div>
  );
};

export default StockGraph;