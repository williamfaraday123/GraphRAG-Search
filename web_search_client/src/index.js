import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';
import SourceView from './SourceView';
import reportWebVitals from './reportWebVitals';

// ?source=<uri>: the full-text page a Sources link opens in a new tab
const source = new URLSearchParams(window.location.search).get('source');

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    {source ? <SourceView uri={source} /> : <App />}
  </React.StrictMode>
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
