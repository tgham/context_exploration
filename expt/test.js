// Import Grid class and practice grids from experiment.js
import { Grid, 
  practice1Grid,
  practice2Grid, 
  practice3Grid,
  practice4Grid, 
  practice5Grid,
  practice6Grid, 
  practice7Grid,
  practice8Grid,
 } from './experiment.js';

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
    question: "What is the difference between ‘tip days’ and ‘toll days’?",
    options: [
      "On tip days, popular intersections earn you a $1 tip; on toll days, busy intersections require you to pay a $1 toll.",
      "There is no difference between tip days and toll days - on all days, you have the chance to receive tips and avoid tolls.",
      "On tip days, you work overtime, meaning you need to select more jobs.",
    ],
    correct: 0
  },
  {
    question: "Suppose you receive the following traffic report. What does this tell you about the objective of your day?",
    options: [
      "You might encounter both tips and tolls today, and so should balance these two together.",
      "Today is a 'tip day', so my aim is to maximise tips.",
      "Today is a 'toll day', so my aim is to minimise tolls.",
    ],
    correct: 2,
    visualContent: function() {
      // Generate vehicle HTML for both contexts
      const vehicleHTMLRow = practice6Grid && practice6Grid.generateDollarRows ? practice6Grid.generateDollarRows('context-row', 'costs') : '';
      const vehicleHTMLColumn = practice6Grid && practice6Grid.generateDollarRows ? practice6Grid.generateDollarRows('context-column', 'costs') : '';
      return `
        <div style="display: flex; justify-content: space-around; gap: 20px; margin: 16px 0 8px 0;">
          <div style="flex: 1; text-align: center;">
              <div class="vehicle-animation-container context-row" style="margin: 8px 0;">
                  <div class="vehicle-display-box context-row">
                      ${vehicleHTMLRow}
                  </div>
              </div>
          </div>
      </div>
      `;
    }
  },
  {
    question: "Suppose it is a tip day, and your taxi has revealed a tip at intersection D7. If you visit that same intersection again later in the same day, what happens?",
    options: [
      "You will definitely receive a tip.",
      "You will definitely not receive a tip.",
      "You might receive a tip, or you might not.",
    ],
    correct: 0,
    visualContent: function() {
      const gridHTML = practice1Grid?.createBlankGridHTML(0, true, false, 'specific', [2, 3], 1) || '';
      return `
        <div style="display: flex; justify-content: center; margin: 0px 0 0px 0;">
          <div style="flex: 1; text-align: center;">
              <div id="grid-container" class="current-job-section" style="zoom: 0.5;">
                ${gridHTML}
              </div>
          </div>
        </div>
      `;
    }
  },
  {
    question: "Suppose it is a toll day. How can previous tolls from earlier in the day help you make smarter decisions later in the same day?",
    options: [
      "If you revisit an intersection that previously didn't required you to pay a toll, you will instead earn a tip the second time.",
      "The toll amount increases if you use the same route multiple times.",
      "You can select jobs that avoid intersections that require you to pay tolls.",
    ],
    correct: 2
  },
  {
    question: "How do you know which intersections might be visited later in the same day?", 
    options: [
      "You can't know which intersections might be visited later in the same day.",
      "They are shown in your upcoming dispatches.",
      "They are shown in your upcoming dispatches, and also highlighted in your current dispatch in pink.",
    ],
    correct: 2,

    visualContent: function() {
      const keyAssignment = Math.random() < 0.5 ? 
            { blue: 'Q', green: 'P' } : 
            { blue: 'P', green: 'Q' };
      const feedback=false;
      const selectedPath=null;
      const firstDay=false;
      const showPink=2;
      const restrictPink=null;
      const showNoPaths=false;
      const trafficReport=false;
      const jobsHTML = practice2Grid.createAllJobsHTML(0, selectedPath, keyAssignment, feedback, firstDay, showPink, restrictPink, showNoPaths, trafficReport).replace(/<div id="cost-message".*?<\/div>/s, '');
      const HTML = `
      <div style="display: flex; justify-content: center; align-items: center; margin: 0px 0 20px 0; width: 100%;">
        <div style="width: 700px; display: flex; justify-content: center;">
          <div class="jobs-layout">
            <div class="upcoming-jobs-container grid">
              <div style="zoom: 0.5;">
                ${jobsHTML}
              </div>
            </div>
          </div>
        </div>
      </div>
      `;

      return HTML;
    }
  },
  {
    question: "Why might it be useful to know about your upcoming dispatches?", // What does it mean if an intersection is highlighted in pink?",
    options: [
      "Because they tell you which intersections might be visited later on in the day.",
      "Because they tell you which intersections are rewarding or costly.",
      "Because they tell you whether it is a column day or a row day.",
    ],
    correct: 0
  },
  {
    question: "What happens to tips or tolls at the start of a new day?",
    options: [
      "They remain in the same locations.",
      "They reset to new locations.",
      "You only keep tip or toll information for the route you chose last."
    ],
    correct: 1
  },
  {
    question: "What is the key difference between 'column days' and 'row days'?",
    options: [
      "In column days, tips are randomly placed, while in row days, they are fixed.",
      "Column days have higher tips overall.",
      "In column days, tips tend to cluster in columns, while in row days, they cluster in rows."
    ],
    correct: 2,
    visualContent: function() {
    // Generate vehicle HTML for both contexts
    const vehicleHTMLRow = practice6Grid && practice6Grid.generateDollarRows ? practice6Grid.generateDollarRows('context-row', 'costs') : '';
    const vehicleHTMLColumn = practice7Grid && practice7Grid.generateDollarRows ? practice7Grid.generateDollarRows('context-column', 'costs') : '';
    return `
      <div style="display: flex; justify-content: space-around; gap: 20px; margin: 16px 0 8px 0;">
      <div style="flex: 1; text-align: center;">
        <div class="vehicle-animation-container context-row" style="margin: 8px 0;">
          <div class="vehicle-display-box context-row">
            ${vehicleHTMLRow}
          </div>
        </div>
      </div>
      <div style="flex: 1; text-align: center;">
        <div class="vehicle-animation-container context-column" style="margin: 8px 0;">
          <div class="vehicle-display-box context-column">
            ${vehicleHTMLColumn}
          </div>
        </div>
      </div>
      </div>
    `;
    }
  },
  {
    question: "Suppose you receive the following traffic report. What does this tell you about how the tips tend to be clustered on that day?",
    options: [
      "It doesn't - the traffic report only indicates whether it's a tip day or a toll day.",
      "Today is a 'row day', so tips (and lack of tips) will tend to be clustered in rows.",
      "Today is a 'column day', so tips (and lack of tips) will tend to be clustered in columns.",
    ],
    correct: 2,
    visualContent: function() {
    // Generate vehicle HTML for both contexts
    const vehicleHTMLRow = practice6Grid && practice6Grid.generateDollarRows ? practice6Grid.generateDollarRows('context-row', 'costs') : '';
    const vehicleHTMLColumn = practice7Grid && practice7Grid.generateDollarRows ? practice7Grid.generateDollarRows('context-column', 'rewards') : '';
    return `
      <div style="display: flex; justify-content: space-around; gap: 20px; margin: 16px 0 8px 0;">
        <div style="flex: 1; text-align: center;">
            <div class="vehicle-animation-container context-row" style="margin: 8px 0;">
                <div class="vehicle-display-box context-column">
                    ${vehicleHTMLColumn}
                </div>
            </div>
        </div>
    </div>
    `;
  }
  },
  {
    question: "Suppose it is 'column + tips day'. You receive the following tip at an intersection. How can this information help you predict other tip locations on that same day?",
    options: [
      "Nothing—intersections are not related to one another.",
      "Other tips are likely to be in the same column (i.e. column G).",
      "This intersection will pay a tip for the rest of the experiment."
    ],
    correct: 1,
    visualContent: function() {
    const gridHTML = practice1Grid?.createBlankGridHTML(0, true, false, 'specific', [5, 6], 1) || '';
    const vehicleHTMLColumn = practice6Grid && practice6Grid.generateDollarRows ? practice6Grid.generateDollarRows('context-column', 'rewards') : '';
    return `
    <div style="display: flex; justify-content: center; margin: 0px 0 0px 0;">
    <div style="flex: 1; text-align: center;">
    <div id="grid-container" class="current-job-section" style="zoom: 0.5;">
    ${gridHTML}
    </div>
    </div>
    </div>
    <div style="display: flex; justify-content: space-around; gap: 20px; margin: 16px 0 8px 0;">
      <div style="flex: 1; text-align: center;">
        <div class="vehicle-animation-container context-column" style="margin: 8px 0;" style="zoom: 0.5;">
          <div class="vehicle-display-box context-column" style="zoom: 0.5;">
            ${vehicleHTMLColumn}
          </div>
        </div>
      </div>
    </div>
    `;
  }

  },
  {
    question: "Suppose it is both a 'toll day' and a 'row day'. You do not pay a toll at an intersection. How can this information help you predict other toll locations on that same day?",
    options: [
      "This intersection will pay a toll for the rest of the experiment.",
      "Nothing—intersections are not related to one another.",
      "The whole row (i.e. row 9) is likely to be free of tolls.",
    ],
    correct: 2,
    visualContent: function() {
    const gridHTML = practice1Grid?.createBlankGridHTML(0, true, false, 'specific', [8,2], 0) || '';
    const vehicleHTMLRow = practice6Grid && practice6Grid.generateDollarRows ? practice6Grid.generateDollarRows('context-row', 'costs') : '';
    return `
    <div style="display: flex; justify-content: center; margin: 0px 0 0px 0;">
    <div style="flex: 1; text-align: center;">
    <div id="grid-container" class="current-job-section" style="zoom: 0.5;">
    ${gridHTML}
    </div>
    </div>
    </div>
    <div style="display: flex; justify-content: space-around; gap: 20px; margin: 16px 0 8px 0;">
      <div style="flex: 1; text-align: center;">
        <div class="vehicle-animation-container context-row" style="margin: 8px 0;" style="zoom: 0.5;">
          <div class="vehicle-display-box context-row" style="zoom: 0.5;">
            ${vehicleHTMLRow}
          </div>
        </div>
      </div>
    </div>
    `;
    }
  },
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
      zoom: 1 !important;
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
      display: flex;
      align-items: center;
      zoom: 1 !important;
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
      type: jsPsychHtmlKeyboardResponse,
      stimulus: function() {
        const visualHTML = questionData.visualContent ? questionData.visualContent() : '';
        const buttonHTML = questionData.options.map((option, idx) => 
          `<button class="quiz-answer" data-index="${idx}">${option}</button>`
        ).join('');
        
        return `
          <div class="quiz-container">
            <div class="quiz-question">Q${questionIndex + 1}: ${questionData.question}</div>
            ${visualHTML}
            <div class="quiz-options-container" id="options-container-${questionIndex}">
              ${buttonHTML}
            </div>
          </div>
        `;
      },
      choices: "NO_KEYS",
      data: {
        question: questionData.question,
        correct_response: questionData.correct,
        question_type: 'quiz'
      },
      on_load: function() {
        const optionsContainer = document.getElementById(`options-container-${questionIndex}`);
        const currentButtons = optionsContainer.querySelectorAll('.quiz-answer');
        let answerSelected = false;
       
        currentButtons.forEach((button) => {
          button.addEventListener('click', function() {
            if (answerSelected) return;
            answerSelected = true;
            
            const selectedIndex = parseInt(button.getAttribute('data-index'));
            
            // Show correct/incorrect feedback
            if (selectedIndex === questionData.correct) {
              button.classList.add('correct');
            } else {
              button.classList.add('incorrect');
              // Highlight the correct answer
              currentButtons[questionData.correct].classList.add('correct');
            }
            
            // Disable all buttons after selection
            currentButtons.forEach(btn => {
              btn.style.pointerEvents = 'none';
              btn.disabled = true;
            });
            
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
                e.preventDefault();
                document.removeEventListener('keydown', proceedListener);
                jsPsych.finishTrial();
              }
            };
            document.addEventListener('keydown', proceedListener);
          });
        });
      },
      trial_duration: null,
      on_finish: function(data) {
        // Track correct answers
        const optionsContainer = document.getElementById(`options-container-${questionIndex}`);
        const currentButtons = optionsContainer.querySelectorAll('.quiz-answer');
        let wasCorrect = false;
        
        currentButtons.forEach((button, index) => {
          if (button.classList.contains('correct') && index === questionData.correct) {
            wasCorrect = true;
          }
        });
        
        if (wasCorrect) {
          const correctCount = jsPsych.data.get().last(1).values()[0]?.correctCount || 0;
          jsPsych.data.addProperties({ correctCount: correctCount + 1 });
        }
        jsPsych.data.addProperties({ total_n_questions: quizQuestions.length });
      }
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
 