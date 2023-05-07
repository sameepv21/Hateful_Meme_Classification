import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Home from './Components/Home';
import Project from './Components/Project';
import Docs from './Components/Docs';
import Navbar from './Components/Navbar'
import Footer from './Components/Footer'

function App() {
  return (
    <Router>
      <div className="App">
        <Navbar />
        <Routes>
          <Route path="/project" element = {<Project />} />
          <Route path="/docs" element = {<Docs />}/>
          <Route exact path="/" element = {<Home />} />
        </Routes>
        <Footer />
      </div>
    </Router>
  );
}

export default App;
