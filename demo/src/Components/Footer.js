import React from "react";

function Footer() {
  return (
    <footer className="bg-gray-800 py-8 justify-between">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 container flex flex-wrap">
        <div className="w-full md:w-1/2 px-4">
          <h5 className="text-white uppercase mb-6 font-bold">Contact Info</h5>
          <p className="text-gray-400 mb-2 leading-relaxed">
            Email: sameep.v@ahduni.edu.in
          </p>
          <p className="text-gray-400 mb-2 leading-relaxed">
            Phone: +91 9998750333
          </p>
          <p className="text-gray-400 mb-2 leading-relaxed">
            Address: C-92, Galaxy Tower, Bodakdev, Ahmedabad, Gujarat, India
          </p>
        </div>
      </div>
    </footer>
  );
}

export default Footer;
