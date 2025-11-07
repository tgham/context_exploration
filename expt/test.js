// Quiz questions and answers
export const quizQuestions = [
  {
    question: "How do you select a taxi job?",
    options: [
      "Click on the passenger icon.",
      "Press the corresponding key on your keyboard.",
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
    ],
    correct: 0
  },
  {
    question: "How do you know which intersections might be visited later in the day?", // What does it mean if an intersection is highlighted in pink?",
    options: [
      // "You will pay a toll if you visit this intersection.",
      // "You have already visited this intersection.",
      // "This intersection appears on one of the jobs you might choose in an upcoming dispatch.",
      "You can't know which intersections might be visited later in the day.",
      "They are shown in your upcoming dispatches.",
      "They are shown in your upcoming dispatches, and also highlighted in your current dispatch in pink.",
    ],
    correct: 2
  },
  {
    question: "How can previous tolls help you make smarter decisions later in the day?",
    options: [
      "If you visit an intersection again, you will not have to pay a toll there.",
      "The toll cost decreases if you use the same route multiple times.",
      "You can avoid jobs that pass through intersections that contain tolls.",
    ],
    correct: 2
  },
  {
    question: "Why might it be useful to know about your upcoming dispatches?", // What does it mean if an intersection is highlighted in pink?",
    options: [
      "Because they tell you which intersections might be visited later on in the day.",
      "Because they tell you which intersections are costly.",
      "Because they tell you whether you are in a row city or a column city.",
    ],
    correct: 0
  },
  {
    question: "What happens to tolls at the start of a new day?",
    options: [
      "They remain in the same locations.",
      "They reset to new locations.",
      "You only keep toll information for the route you chose last."
    ],
    correct: 1
  },
  {
    question: "What happens after working in the same city for 5 days?",
    options: [
      "You continue working in the same city, but with new toll locations.",
      "You move to a new city, with a new background and a potentially different traffic pattern.",
      "The toll costs become permanently visible.",
    ],
    correct: 1
  },
  {
    question: "What is the key difference between 'column cities' and 'row cities'?",
    options: [
      "In column cities, tolls are randomly placed, while in row cities, they are fixed.",
      "Column cities have higher tolls overall.",
      "In column cities, tolls tend to cluster in columns, while in row cities, they cluster in rows."
    ],
    correct: 2
  },
  {
    question: "How can you figure out what type of city you are in (i.e. a column city or a row city)?",
    options: [
      "By checking the city background color.",
      "By noticing whether tolls tend to be clustered in columns or rows.",
      "By counting how many jobs you complete each day.",
    ],
    correct: 1
  },
  {
    question: "Suppose you observe a toll at an intersection. How can this information help you predict other toll locations on that day?",
    options: [
      "Nothing—intersections are not related to one another.",
      "Other tolls are likely to be in the same row or column, depending on the city pattern.",
      "This intersection will have a toll for the rest of the experiment."
    ],
    correct: 1
  },
  {
    question: "At the start of a new day within the same city, what pattern will the new set of tolls have?",
    options: [
      "The same pattern as previous days in that city (i.e. clustered in columns or rows).",
      "A completely random pattern with no relation to previous days.",
      "A pattern that depends on the jobs you selected.",
    ],
    correct: 0
  }
 ];
 
 // Function to create quiz trials
 export function createQuizTrials(jsPsych) {
  const quizTrials = [];
   
  // Add CSS styles for vertical stacking with fixed width buttons
  const styleElement = document.createElement('style');
  styleElement.textContent = `
    .quiz-container {
      width: 100%;
      max-width: 1000px;
      margin: 0 auto;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 20px;
    }
   
    .quiz-question {
      font-size: 28px;
      margin-bottom: 40px;
      text-align: center;
      width: 90%;
    }
   
    .quiz-options-container {
      width: 700px;
      display: flex;
      flex-direction: column;
      align-items: center;
    }
   
    .jspsych-html-button-response-button {
      margin: 8px 0;
      width: 100%;
    }
   
    .quiz-answer {
      width: 100%;
      min-height: 70px;
      text-align: left;
      padding: 15px 25px;
      margin: 8px 0;
      border: 1px solid #ccc;
      border-radius: 5px;
      background-color: #f9f9f9;
      font-size: 22px;
      cursor: pointer;
      transition: background-color 0.3s;
      display: flex;
      align-items: center;
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
      margin-top: 40px;
      text-align: center;
      font-size: 22px;
    }
   
    .quiz-section {
      width: 90%;
      max-width: 800px;
      margin: 0 auto;
      text-align: center;
    }
   
    .quiz-section h2 {
      font-size: 32px;
      margin-bottom: 20px;
    }
   
    .quiz-section p, .quiz-section li {
      font-size: 22px;
      line-height: 1.5;
    }
   
    .quiz-section ul {
      text-align: left;
      margin: 20px auto;
      max-width: 600px;
    }
  `;
  document.head.appendChild(styleElement);
   // Welcome screen
  const welcomeTrial = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: `
      <div class="instruction-section">
        <h2>Knowledge Check Quiz</h2>
        <p>You will be presented with 11 questions about the Taxi Coordination task.</p>
        <p>For each question:</p>
        <p>
          <p>- Click on the answer you believe is correct.</p>
          <p>- Correct answers will turn green, and incorrect answers will turn red.</p>
          <p>- After selecting an answer, press the spacebar to proceed to the next question.</p>
        </p>
      </div>
      <div class="instruction-section">
        <h2>Press the spacebar to begin the quiz.</h2>
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
          <div class="quiz-options-container" id="options-container-${questionIndex}"></div>
        </div>
      `,
      choices: questionData.options.map(option => option),
      button_html: '<button class="quiz-answer">%choice%</button>',
      data: {
        question: questionData.question,
        correct_response: questionData.correct,
        question_type: 'quiz'
      },
      on_load: function() {
        // Move buttons into options container for better positioning
        const optionsContainer = document.getElementById(`options-container-${questionIndex}`);
        const buttons = document.querySelectorAll('.jspsych-html-button-response-button');
       
        buttons.forEach((button) => {
          optionsContainer.appendChild(button);
        });
       
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
            jsPsych.data.addProperties({ total_n_questions: quizQuestions.length });
           
            // Add instruction to press spacebar
            const quizContainer = document.querySelector('.quiz-container');
            const nextInstructions = document.createElement('div');
            nextInstructions.classList.add('spacebar-container');
            nextInstructions.innerHTML = `
              <div>Press spacebar to continue to the next question.</div>
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

  // const finalTrial = {
  //   type: jsPsychHtmlKeyboardResponse,
  //   stimulus: function() {
  //     // Retrieve the correct answer count
  //     const correctCount = jsPsych.data.get().last(1).values()[0]?.correctCount || 0;
  //     const percentage = Math.round((correctCount / quizQuestions.length) * 100);
  //     const passed = percentage >= 70;
  //     jsPsych.data.addProperties({ quiz_passed: passed });
      
  //     return `
  //       <div class="instruction-section">
  //         <h2>Quiz Complete!</h2>
  //         <p>You answered ${correctCount} out of ${quizQuestions.length} questions correctly (${percentage}%).</p>
  //         ${passed
  //           ? '<p>Congratulations! You passed the quiz. Press the spacebar to continue with the experiment.</p>'
  //           : '<p>Unfortunately, you did not pass the quiz. Please return to Prolific.</p>'}
  //       </div>
  //     `;
  //   },
  //   choices: function() {
  //     const correctCount = jsPsych.data.get().last(1).values()[0]?.correctCount || 0;
  //     const percentage = Math.round((correctCount / quizQuestions.length) * 100);
  //     return percentage >= 70 ? [' '] : 'NO_KEYS';
  //   },
  //   trial_duration: function() {
  //     const correctCount = jsPsych.data.get().last(1).values()[0]?.correctCount || 0;
  //     const percentage = Math.round((correctCount / quizQuestions.length) * 100);
  //     // If failed, show message for 2s then end trial to trigger on_finish
  //     return percentage >= 70 ? null : 2000;
  //   },
  //   on_finish: function() {
  //     const passed = jsPsych.data.get().last(1).values()[0]?.quiz_passed;
  //     if (!passed) {
  //       const ppt_data = jsPsych.data.get().json();
  //       send_complete(id, ppt_data)
  //         .catch(error => {
  //           console.error('Failed to send completion data:', error);
  //         })
  //         .finally(() => {
  //           window.location.replace("https://app.prolific.com/submissions/complete?cc=C37PLZK3");
  //         });
  //     }
  //   }
  // };
 
 
  // quizTrials.push(finalTrial);
   return quizTrials;
 }
 