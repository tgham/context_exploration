// Quiz questions and answers
export const quizQuestions = [
  {
    question: "How do you select a taxi job?",
    options: [
      "Click on the passenger icon.",
      "Press the corresponding key on your keyboard.",
      "Drag and drop the taxi onto the route.",
      "Wait for the system to assign a job automatically."
    ],
    correct: 1 // Index of correct answer (0-based)
  },
  {
    question: "If your taxi reveals a toll at an intersection, and you visit that intersection again later in the day, what happens?",
    options: [
      "You will definitely receive a toll.",
      "You will definitely not receive a toll.",
      "You might receive a toll, or you might not.",
      "Impossible, because your taxi can't travel via intersections that have already been visited."
    ],
    correct: 0
  },
  // {
  //   question: "How can previous tolls help you make smarter decisions later in the day?",
  //   options: [
  //     "You can avoid jobs that pass through expensive intersections.",
  //     "You can remove tolls by revisiting intersections.",
  //     "The toll cost decreases if you use the same route multiple times.",
  //     "You can see all toll locations before making a choice."
  //   ],
  //   correct: 0
  // },
  // {
  //   question: "Where can you see information about the jobs you will need to choose between later in the day?",
  //   options: [
  //     "You can't - you can only see your current job.",
  //     "In a separate menu that you need to click to access.",
  //     "At the top of the screen, above your current job.",
  //     "Below your current job, in the upcoming jobs section."
  //   ],
  //   correct: 3
  // },
  // {
  //   question: "What happens to the tolls at the end of the day?",
  //   options: [
  //     "They remain in the same locations.",
  //     "They disappear permanently.",
  //     "They reset to new locations.",
  //     "You only keep toll information for the route you chose last."
  //   ],
  //   correct: 2
  // },
  // {
  //   question: "What happens after working in the same city for 4 days?",
  //   options: [
  //     "You continue working in the same city, but with new toll locations.",
  //     "You move to a new city, with a new background and a potentially different traffic pattern.",
  //     "The toll costs become permanently visible.",
  //     "The number of jobs you manage per day increases."
  //   ],
  //   correct: 1
  // },
  // {
  //   question: "What is the key difference between 'column cities' and 'row cities'?",
  //   options: [
  //     "In column cities, tolls are randomly placed, while in row cities, they are fixed.",
  //     "Column cities have higher tolls overall.",
  //     "Row cities reset their tolls less often.",
  //     "In column cities, tolls tend to cluster in columns, while in row cities, they cluster in rows."
  //   ],
  //   correct: 3
  // },
  // {
  //   question: "How can you figure out what type of city you are in (i.e. a column city or a row city)?",
  //   options: [
  //     "By checking the city background color.",
  //     "By noticing whether tolls tend to appear in columns or rows.",
  //     "By counting how many jobs you complete each day.",
  //     "By comparing today's tolls to yesterday's tolls."
  //   ],
  //   correct: 1
  // },
  // {
  //   question: "Suppose you observe a toll at an intersection. How can this information help you predict other toll locations?",
  //   options: [
  //     "Nothing—tolls are randomly placed each day.",
  //     "Other tolls are likely to be in the same row or column, depending on the city pattern.",
  //     "The next intersection you visit will definitely have a toll.",
  //     "This intersection will have a toll for the rest of the experiment."
  //   ],
  //   correct: 1
  // },
  // {
  //   question: "At the end of each day within a city, what pattern will the new set of tolls have?",
  //   options: [
  //     "A completely random pattern with no relation to previous days.",
  //     "The same pattern as previous days (clustered in columns or rows).",
  //     "A pattern that depends on the jobs you selected.",
  //     "A mix of both column and row clustering."
  //   ],
  //   correct: 1
  // }
];

// Function to create quiz trials
export function createQuizTrials(jsPsych) {
  const quizTrials = [];
    
  // Add CSS styles for vertical stacking
  const styleElement = document.createElement('style');
  styleElement.textContent = `
    .quiz-container {
      max-width: 800px;
      margin: 0 auto;
      display: flex;
      flex-direction: column;
      justify-content: center;
      height: 20vh; /* Center content vertically */
    }
    .quiz-question {
      font-size: 28px; /* Increased font size */
      margin-bottom: 40px; /* Larger vertical margin */
      text-align: center; /* Center-aligned text */
    }
    .jspsych-html-button-response-button {
      display: block;
      margin: 20px auto; /* Larger vertical margin */
      width: 100%;
    }
    .quiz-answer {
      display: block;
      width: 100%;
      text-align: left;
      padding: 25px; /* Increased padding */
      margin-top: 10px; /* Larger vertical margin */
      margin-bottom: 10px; /* Larger vertical margin */
      border: 1px solid #ccc;
      border-radius: 5px;
      background-color: #f9f9f9;
      font-size: 22px; /* Increased font size */
      cursor: pointer;
      transition: background-color 0.3s;
    }
    .quiz-answer:hover {
      background-color: #e9e9e9;
    }
    .correct {
      background-color: #dff0d8 !important;
      border-color: #d6e9c6 !important;
      color: #3c763d !important;
    }
    .incorrect {
      background-color: #f2dede !important;
      border-color: #ebccd1 !important;
      color: #a94442 !important;
    }
    .spacebar-container {
      margin-top: 40px; /* Larger vertical margin */
      text-align: center;
      font-size: 22px; /* Increased font size */
    }
  `;
  document.head.appendChild(styleElement);
  
  // Welcome screen
  const welcomeTrial = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: `
      <div class="instruction-section">
        <h2>Knowledge Check Quiz</h2>
        <p>You will be presented with 10 questions about the taxi coordination task.</p>
        <p>For each question:</p>
        <ul>
          <li>Click on the answer you believe is correct.</li>
          <li>Correct answers will turn green, and incorrect answers will turn red.</li>
          <li>After selecting an answer, press the spacebar to proceed to the next question.</li>
        </ul>
        <p>Press the spacebar to begin the quiz.</p>
      </div>
    `,
    choices: [' ']
  };
  
  quizTrials.push(welcomeTrial);
  
  // Create a trial for each question
  quizQuestions.forEach((questionData, questionIndex) => {
    quizTrials.push({
      type: jsPsychHtmlButtonResponse,
      stimulus: `
        <div class="quiz-container">
          <div class="quiz-question">Question ${questionIndex + 1}: ${questionData.question}</div>
        </div>
      `,
      choices: questionData.options.map(option => option), // Explicitly map each option
      button_html: '<button class="quiz-answer">%choice%</button>', // Use %choice% template
      data: {
        question: questionData.question,
        correct_response: questionData.correct,
        question_type: 'quiz'
      },
      on_load: function() {
        // Add event listeners to answer buttons
        document.querySelectorAll('.quiz-answer').forEach((button, index) => {
          button.addEventListener('click', function() {
            // Check if correct and add appropriate class
            if (index === questionData.correct) {
              button.classList.add('correct');
              // Increment correct answer counter
              const correctCount = jsPsych.data.get().last(1).values()[0]?.correctCount || 0;
              jsPsych.data.addProperties({ correctCount: correctCount + 1 });
            } else {
              button.classList.add('incorrect');
              // Highlight the correct answer
              document.querySelectorAll('.quiz-answer')[questionData.correct].classList.add('correct');
            }
            
            // Add instruction to press spacebar
            const quizContainer = document.querySelector('.quiz-container');
            const nextInstructions = document.createElement('div');
            nextInstructions.classList.add('instruction-section');
            nextInstructions.innerHTML = `
              <div style="margin-top: 20px;">Press spacebar to continue to the next question.</div>
            `;
            quizContainer.appendChild(nextInstructions);
            
            // Set up spacebar listener to progress
            const proceedListener = function(e) {
              if (e.code === 'Space') {
                document.removeEventListener('keydown', proceedListener);
                jsPsych.finishTrial();
              }
            };
            document.addEventListener('keydown', proceedListener);
            
            // Disable all buttons after selection
            document.querySelectorAll('.quiz-answer').forEach(btn => {
              btn.style.pointerEvents = 'none';
            });
          }, { once: true });
        });
      },
      trial_duration: null,
      response_ends_trial: false
    });
  });
  
  // Final feedback screen
  const finalTrial = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {
      // Retrieve the correct answer count
      const correctCount = jsPsych.data.get().last(1).values()[0]?.correctCount || 0;
      const percentage = Math.round((correctCount / quizQuestions.length) * 100);
      const passed = percentage >= 60;
      
      return `
        <div class="instruction-section">
          <h2>Quiz Complete!</h2>
          <p>You answered ${correctCount} out of ${quizQuestions.length} questions correctly (${percentage}%).</p>
          ${passed 
            ? '<p>Congratulations! You passed the quiz. Press the spacebar to continue with the experiment.</p>' 
            : '<p>Unfortunately, you did not pass the quiz. Please review the instructions and try again.</p>'}
        </div>
      `;
    },
    choices: function() {
      // Allow spacebar only if participant passed
      const correctCount = jsPsych.data.get().last(1).values()[0]?.correctCount || 0;
      const percentage = Math.round((correctCount / quizQuestions.length) * 100);
      return percentage >= 60 ? [' '] : [];
    }
  };
  
  quizTrials.push(finalTrial);
  
  return quizTrials;
}