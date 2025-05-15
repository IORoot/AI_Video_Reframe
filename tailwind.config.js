/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./index.html",
    "./**/*.{js,ts,jsx,tsx}",
    "!./node_modules/**/*"
  ],
  theme: {
    extend: {
      colors: {
        // You can add custom colors here
      },
    },
  },
  plugins: [],
} 