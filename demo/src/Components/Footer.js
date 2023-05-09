import React from "react";

function Footer() {
    let github_url = "https://github.com/sameepv21/Hateful_Meme_Classification"
    let report_url = "https://drive.google.com/file/d/1U_tDC7jeNkLZUwaJ8fguL8b1jVgZFYgk/view?usp=sharing"
    let poster_url = "https://drive.google.com/file/d/1wiIA0t5FPhAJ-aH_i9TTLn_yi6s_PPEz/view?usp=sharing"
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
        <div className="w-full md:w-1/2 px-4 text-right">
          <h5 className="text-white uppercase mb-6 font-bold">Links</h5>
          <ul className="list-unstyled">
            <li>
              <a href={report_url} className="text-gray-400 hover:text-white font-semibold block pb-2 text-sm">
                Project Report
              </a>
            </li>
            <li>
              <a href={poster_url} className="text-gray-400 hover:text-white font-semibold block pb-2 text-sm">
                Presentation
              </a>
            </li>
            <li>
              <a href={github_url} className="text-gray-400 hover:text-white font-semibold block pb-2 text-sm">
                Code
              </a>
            </li>
          </ul>
        </div>
      </div>
    </footer>
  );
}

export default Footer;