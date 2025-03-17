// Initialize jsPsych
const jsPsych = initJsPsych({
    on_finish: function() {
      jsPsych.data.displayData();
    }
  });
  
  // Define path data - you would replace this with your CSV loading logic
  const bluePath = [
    {row: 0, col: 3, marker: 'S', cost: false},
    {row: 1, col: 3, marker: '★', cost: true},
    {row: 2, col: 3, marker: '★', cost: false},
    {row: 3, col: 3, marker: '★', cost: true},
    {row: 4, col: 3, marker: '★', cost: false},
    {row: 5, col: 3, marker: '★', cost: true},
    {row: 6, col: 3, marker: '★', cost: false},
    {row: 7, col: 3, marker: '★', cost: true},
    {row: 8, col: 3, marker: '★', cost: false},
    {row: 9, col: 3, marker: 'G', cost: false}
  ];
  
  const redPath = [
    {row: 8, col: 1, marker: 'S', cost: false},
    {row: 8, col: 2, marker: '★', cost: false},
    {row: 8, col: 3, marker: '★', cost: true},
    {row: 8, col: 4, marker: '★', cost: false},
    {row: 8, col: 5, marker: '★', cost: true},
    {row: 8, col: 6, marker: '★', cost: false},
    {row: 8, col: 7, marker: '★', cost: true},
    {row: 8, col: 8, marker: '★', cost: false},
    {row: 7, col: 8, marker: '★', cost: true},
    {row: 6, col: 8, marker: 'G', cost: false}
  ];
  
  // Function to generate initial grid HTML
  function createGridHTML() {
    let gridHTML = '<div class="grid-container">';

    // Create empty 11x11 grid
    for (let row = 0; row < 11; row++) {
        for (let col = 0; col < 11; col++) {
            // Check if this cell is part of blue path
            const blueCell = bluePath.find(cell => cell.row === row && cell.col === col);
            // Check if this cell is part of red path
            const redCell = redPath.find(cell => cell.row === row && cell.col === col);

            if (blueCell) {
                gridHTML += `<div class="grid-cell blue-path" id="cell-${row}-${col}">${blueCell.marker}</div>`;
            } else if (redCell) {
                gridHTML += `<div class="grid-cell red-path" id="cell-${row}-${col}">${redCell.marker}</div>`;
            } else {
                gridHTML += `<div class="grid-cell" id="cell-${row}-${col}"></div>`;
            }
        }
    }

    gridHTML += '</div>';

    // Add choice options
    gridHTML += `
      <div class="choice-container">
        <div class="choice-box" style="background-color: blue;">Blue</div>
        <div class="choice-box" style="background-color: red;">Red</div>
      </div>
      <div style="text-align: center; margin-top: 20px;">
        Press left arrow to choose blue path, right arrow to choose red path
      </div>
    `;

    return gridHTML;
}
  
  // Function to animate the agent along the chosen path
  function animateAgent(path, pathColor, callback) {
    let currentStep = 0;
    const totalSteps = path.length;

    // Clear the entire grid
    document.querySelectorAll('.grid-cell').forEach(cell => {
        cell.textContent = ''; // Clear the content
        cell.classList.remove('blue-path', 'red-path', 'avatar'); // Remove all styling
        cell.style.backgroundColor = 'white'; // Reset background color
    });

    // Replot the chosen path in its respective color
    path.forEach(cell => {
        const cellElement = document.getElementById(`cell-${cell.row}-${cell.col}`);
        if (cell.marker === 'S' || cell.marker === 'G') {
            cellElement.textContent = cell.marker; // Show start (S) or goal (G)
        } else {
            cellElement.textContent = '★'; // Show path marker
        }
        cellElement.classList.add(pathColor); // Add the chosen path color class
        cellElement.style.backgroundColor = ''; // Reset background color to default for the path
    });

    // Animation function
    function step() {
        if (currentStep > 0) {
            // Mark previous cell as visited with cost indicator
            const prevCell = path[currentStep - 1];
            const prevCellElement = document.getElementById(`cell-${prevCell.row}-${prevCell.col}`);
            prevCellElement.classList.remove('avatar');

            if (prevCell.cost) {
                prevCellElement.innerHTML += ' <span class="cost">$</span>';
                prevCellElement.style.backgroundColor = 'red'; // Mark as red if cost
            } else {
                prevCellElement.innerHTML += ' <span class="no-cost">✓</span>';
                prevCellElement.style.backgroundColor = 'grey'; // Mark as grey if no cost
            }
        }

        if (currentStep < totalSteps) {
            // Move agent to current cell
            const currentCell = path[currentStep];
            const cellElement = document.getElementById(`cell-${currentCell.row}-${currentCell.col}`);
            cellElement.classList.add('avatar');

            currentStep++;
            setTimeout(step, 1000); // 1 second per step
        } else {
            // Animation complete
            if (callback) callback();
        }
    }

    // Start animation
    setTimeout(step, 500); // Start after a short delay
}
  
  // Instructions
  const instructions = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: `
      <h2>Path Selection Task</h2>
      <p>In this experiment, you will see a grid with two colored paths.</p>
      <p>Each path has a start point (S) and a goal point (G).</p>
      <p>Your task is to choose one of these paths using the arrow keys:</p>
      <p>- Press the <strong>left arrow</strong> to choose the blue path</p>
      <p>- Press the <strong>right arrow</strong> to choose the red path</p>
      <p>After you choose, you'll see an avatar move along the path.</p>
      <p>Some states will have a cost ($), and others will be free (✓).</p>
      <p>Press any key to begin.</p>
    `,
    choices: "ALL_KEYS"
  };
  
  // Path selection trial
  const pathSelectionTrial = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {
      return createGridHTML();
    },
    choices: ['ArrowLeft', 'ArrowRight'],
    prompt: "",
    data: {
      task: 'path_selection'
    },
    on_finish: function(data) {
      if (data.response === 'ArrowLeft') {
        data.choice = 'blue';
      } else if (data.response === 'ArrowRight') {
        data.choice = 'red';
      }
    }
  };
  
  // Path animation trial
  const pathAnimationTrial = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {
        return createGridHTML();
    },
    choices: "NO_KEYS",
    trial_duration: null, // Will be set dynamically
    on_load: function() {
        const last_trial = jsPsych.data.get().last(1).values()[0];
        const chosenPath = last_trial.choice === 'blue' ? bluePath : redPath;
        const pathColor = last_trial.choice === 'blue' ? 'blue-path' : 'red-path';

        // Calculate animation duration (1s per step + buffer)
        const animationDuration = (chosenPath.length * 1000) + 2000;

        // Set trial duration correctly
        jsPsych.getCurrentTrial().trial_duration = animationDuration;

        // Start animation
        animateAgent(chosenPath, pathColor, function() {
            // Animation complete, end trial after a short delay
            setTimeout(function() {
                jsPsych.finishTrial();
            }, 1500);
        });
    },
    data: function() {
        const last_trial = jsPsych.data.get().last(1).values()[0];
        return {
            choice: last_trial.choice
        };
    }
};

  
  // Results summary
  const resultsSummary = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {
      const last_trial = jsPsych.data.get().last(2).values()[0]; // Get the path selection trial
      const chosenPath = last_trial.choice === 'blue' ? bluePath : redPath;
      
      // Calculate total cost
      const totalCost = chosenPath.filter(cell => cell.cost).length;
      
      return `
        <h2>Results Summary</h2>
        <p>You chose the ${last_trial.choice} path.</p>
        <p>Total cost: ${totalCost} states</p>
        <p>Press any key to continue.</p>
      `;
    },
    choices: "ALL_KEYS"
  };
  
  // End message
  const end = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: `
      <p>Thank you for participating!</p>
      <p>The experiment is now complete.</p>
      <p>Press any key to see your data.</p>
    `,
    choices: "ALL_KEYS"
  };
  
  // Create timeline
  const timeline = [
    instructions,
    pathSelectionTrial,
    pathAnimationTrial,
    resultsSummary,
    end
  ];
  
  // Start experiment when the page loads
  window.onload = function() {
    jsPsych.run(timeline);
  };