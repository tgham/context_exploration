// Initialize jsPsych
const jsPsych = initJsPsych({
    on_finish: function() {
        var ppt_data = jsPsych.data.get().json();
        send_complete(subject_id, ppt_data);
        console.log('experiment complete');
        window.location.replace("error.html"); // REPLACE WITH PROLIFIC URL
    }
});

import { createQuizTrials } from './test.js';
document.body.style.zoom = "100%";

// decide whether we're doing this properly or not...
let test = false;
let subject_id = null;
let sequence = null;
let data = null;
let grid = null;
let currentTrialIndex = 0;

// just test with this...
if (test) {

    // var subject_id = 1
    jsPsych.data.addProperties({
        subject_id: subject_id,
    });
    var ppt_data = jsPsych.data.get().json();
    send_incomplete(subject_id, ppt_data);
    console.log('debugging with subject_id 1');
    fetch('assets/trial_sequences/expt_info_1.json')
    .then(response => response.json())
    .then(data => {
        grid = new Grid(data); // Initialize the Grid class with the loaded data        
        initializeExperiment(); // Call a function to start the experiment
        console.log('loaded grid')
    })
    .catch(error => console.error('Error loading JSON:', error));

    // rename sequence to data, and then use this to generate the grid
    let data;
    data = sequence;
    grid = new Grid(data); // Initialize the Grid class with the loaded data
    const numCities = data.env_costs.n_cities; // Assuming this is the number of cities
    createCityMapping(numCities);
    console.log('Grid data loaded:', grid);
    console.log('City mapping created:', cityMapping);

} else {

    // capture info from Prolific and fetch ID from backend. If null, then redirect to error page
    var pid = get_prolific_id();
    console.log('PID:',pid)
    create_participant(pid).then((value) => {
        if (value['id'] == null) {
            console.error(`${pid} is not unique or an error occurred.`);
            // window.location.replace("error.html");
            return;
        }
        subject_id = value['id'];
        data = value['sequence'];
        console.log(`id => ${subject_id}`);
        console.log(`sequence => ${data}`);
        grid = new Grid(data); // Initialize the Grid class with the loaded data        
        initializeExperiment(); // Call a function to start the experiment
    }).catch((error) => {
        console.error('Failed to fetch participant ID:', error);
        // window.location.replace("error.html");
    });
    console.log('loaded PID etc.')
}

// get sound ready
const audioContext = new (window.AudioContext || window.webkitAudioContext)();
let costSoundBuffer;
fetch('assets/costSound.mp3')
    .then(response => response.arrayBuffer())
    .then(data => audioContext.decodeAudioData(data))
    .then(buffer => {
        costSoundBuffer = buffer;
    });

function playCostSound() {
    if (costSoundBuffer) {
        const source = audioContext.createBufferSource();
        source.buffer = costSoundBuffer;
        source.connect(audioContext.destination);
        source.start(0); // Play the sound
    }
}

// Define the Grid class
class Grid {
    constructor(gridData) {
        this.trialInfo = gridData.trial_info; // Array of trial info
        this.envCosts = gridData.env_costs; // Object of environment costs
        this.gridSize = gridData.env_costs.grid_size; // Grid size (N)
        this.nTrials = gridData.env_costs.n_trials
        this.nGrids = gridData.env_costs.n_grids
        this.nCities = gridData.env_costs.n_cities
        this.observedCosts = {}; // Track observed costs for each grid
        for (let i = 0; i < this.nTrials; i++) {
            this[`observedCosts${i}`] = {};
        }
        this.currentGrid = 0; // Track the current grid
        this.currentCity = null; // Track the current city

    }

    // Get the binary costs for a specific grid
    getBinaryCosts(gridId) {
        return this.envCosts[gridId];
    }

    // Get the trial info for a specific trial
    getTrialInfo(trialIndex) {
        return this.trialInfo[trialIndex];
    }

    // Get the start and goal positions for a specific trial
    getStartAndGoal(trialIndex) {
        const trial = this.getTrialInfo(trialIndex);
        return {
            startA: trial.start_A,
            startB: trial.start_B,
            goalA: trial.goal_A,
            goalB: trial.goal_B
        };
    }

    // Get the paths for a specific trial
    getPaths(trialIndex) {
        const trial = this.getTrialInfo(trialIndex);
        return {
            pathA: trial.path_A,
            pathB: trial.path_B
        };
    }

    // Create the grid HTML for a specific trial
    createGridHTML = function(trialIndex, selectedPath = null, keyAssignment = null, includeCostDisplay = true, practice=false, feedback=false) {
        const trial = this.getTrialInfo(trialIndex);
        const city = trial.city;
        const grid = trial.grid;
        const binaryCosts = this.getBinaryCosts(`city_${city}_grid_${grid}`);
        const gridSize = this.gridSize;
        const jobNumber = (trialIndex % this.nTrials) + 1; // Job number within the grid
        
        let gridHTML = `
            <div class="current-job-container">
        `;

        if (includeCostDisplay) {
            // let currentDay = this.currentGrid; 
            if (!practice) {
            gridHTML += `
            <div class="cost-display-container">
                <h2 class="day-display">Day ${trial.grid}/${this.nGrids}</h2>
                <h2 class="cost-total">Total Tolls Paid Today:</h2>
                <p id="total-cost" class="cost-total.">-$${totalCost}</p>
                <p id="trial-cost" class="cost-trial hidden">-$0</p> 
            </div>
            `;
            } else {
            gridHTML += `
            <div class="cost-display-container">
                <h2 class="day-display">Practice Day ${trial.grid}/${this.nGrids}</h2>
                <h2 class="cost-total">Total Tolls Paid:</h2>
                <p id="total-cost" class="cost-total">-$${totalCost}</p>
                <p id="trial-cost" class="cost-trial hidden">-$0</p> 
            </div>
            `;
            }
        }   
        

        if (feedback) {
            gridHTML += `
            <div class="cost-display-container">
                <h2>You paid <strong style="color: #f87171;">$${totalCost}</strong> today.</h2>
                <p id="trial-cost" class="cost-trial hidden">-$0</p> 
                <p id="total-cost" class="cost-total">A new day has begun.</p>
                <p id="total-cost" class="cost-total">Tolls in this city have been reset.</p>
            </div>
            `;
        }

        gridHTML += `
                <div class="grid-container" style="grid-template-columns: repeat(${gridSize}, 40px); background-color: #ece75d;">
        `;
    
        for (let row = 0; row < gridSize; row++) {
            for (let col = 0; col < gridSize; col++) {
                // const cellId = `cell-${row}-${col}`;
                const cellId = `cell-${row}-${col}-trial-${trial.trial}`;
                const isStartA = selectedPath !== 'green' && selectedPath !== 'none' && row === trial.start_A[0] && col === trial.start_A[1];
                const isStartB = selectedPath !== 'blue' && selectedPath !== 'none' && row === trial.start_B[0] && col === trial.start_B[1];
                const isGoalA = selectedPath !== 'green' && selectedPath !== 'none' && row === trial.goal_A[0] && col === trial.goal_A[1];
                const isGoalB = selectedPath !== 'blue' && selectedPath !== 'none' && row === trial.goal_B[0] && col === trial.goal_B[1];
                const isPathA = selectedPath !== 'green' && selectedPath !== 'none' && trial.path_A.some(coord => coord[0] === row && coord[1] === col);
                const isPathB = selectedPath !== 'blue' && selectedPath !== 'none' && trial.path_B.some(coord => coord[0] === row && coord[1] === col);
    
                const observedCost = this.observedCosts[`${row}-${col}`];
                const observedClass = observedCost !== undefined ? 
                    (observedCost === -1 ? 'observed-cost' : 'observed-no-cost') : '';
    
                // Handle overlapping paths
                const isOverlap = isPathA && isPathB;
                let pathClass = '';
                let content = ''; // Content for the cell (e.g., letter or other marker)
                
                // Determine content based on key assignment if provided
                if (keyAssignment) {
                    if (isOverlap) {

                        // for simplicity, let's just keep it consistent
                        pathClass = 'blue-path';
                        content = `<span class="green-text">⚝</span>`;

                        // random 
                        // const randomChoice = Math.random() < 0.5;
                        // pathClass = randomChoice ? 'blue-path' : 'green-path';
                        // content = randomChoice ? `<span class="green-text">⚝</span>` : `<span class="blue-text">⚝</span>`;

                    } else if (isPathA) {
                        pathClass = 'blue-path';
                        content = keyAssignment.blue;
                    } else if (isPathB) {
                        pathClass = 'green-path';
                        content = keyAssignment.green;
                    }
                } else {
                    // Fall back to stars if no key assignment provided
                    if (isOverlap) {
                        const randomChoice = Math.random() < 0.5;
                        pathClass = randomChoice ? 'blue-path' : 'green-path';
                        content = randomChoice ? '<span class="green-text">⚝</span>' : '<span class="blue-text">⚝</span>';
                    } else if (isPathA) {
                        pathClass = 'blue-path';
                        content = '⚝';
                    } else if (isPathB) {
                        pathClass = 'green-path';
                        content = '⚝';
                    }
                }
    
                if (isStartA) {
                    gridHTML += `<div class="grid-cell start blue-path ${observedClass}" id="${cellId}">
                                    <img src="assets/people/blue_person.png" alt="Blue Start" width="30" height="30">
                                 </div>`;
                } else if (isStartB) {
                    gridHTML += `<div class="grid-cell start green-path ${observedClass}" id="${cellId}">
                                    <img src="assets/people/green_person.png" alt="Green Start" width="30" height="30">
                                 </div>`;
                } else if (isGoalA) {
                    gridHTML += `<div class="grid-cell goal blue-path ${observedClass}" id="${cellId}">🏠</div>`;
                } else if (isGoalB) {
                    gridHTML += `<div class="grid-cell goal green-path ${observedClass}" id="${cellId}">🏠</div>`;
                } else if (isPathA || isPathB || isOverlap) {
                    gridHTML += `<div class="grid-cell ${observedClass} ${pathClass}" id="${cellId}" style="font-size: 2rem;">${content}</div>`;
                } else {
                    gridHTML += `<div class="grid-cell ${observedClass}" id="${cellId}"></div>`;
                }
            }
        }   
        gridHTML += `</div></div>`;
    
        return gridHTML;
    };

    // method for plotting a grid, either blank or with all costs revealed
    createBlankGridHTML(trialIndex = null, revealCosts = false, feedback=false) {

        let gridHTML = `
            <div class="current-job-container">
        `;
        
        if (feedback){
            gridHTML += `
            <div class="cost-display-container">
                <h2>You paid <strong style="color:  #f87171;">$${totalCost}</strong> today.</h2>
                <p id="trial-cost" class="cost-trial hidden">-$0</p> 
                <p id="total-cost" class="cost-total">A new day has begun.</p>
                <p id="total-cost" class="cost-total">Tolls in this city have been reset.</p>
            </div>
            `;
        } 

        gridHTML += `
            <div class="grid-container" style="grid-template-columns: repeat(${this.gridSize}, 40px);">
        `;
    
        let binaryCosts = null; // Initialize binaryCosts
    
        if (trialIndex !== null) {
            const trial = this.getTrialInfo(trialIndex);
            const city = trial.city;
            const grid = trial.grid;
    
            // Attempt to get binaryCosts
            binaryCosts = this.getBinaryCosts(`city_${city}_grid_${grid}`);
            if (!binaryCosts) {
                console.warn(`binaryCosts is undefined for city_${city}_grid_${grid}`);
                binaryCosts = Array.from({ length: this.gridSize }, () => Array(this.gridSize).fill(0)); // Default grid
            }
        }
    
        for (let row = 0; row < this.gridSize; row++) {
            for (let col = 0; col < this.gridSize; col++) {
                const cellId = `cell-${row}-${col}`;
    
                if (revealCosts) {
                    const cost = binaryCosts ? binaryCosts[row][col] : 0; // Safely access binaryCosts
                    const costClass = cost === -1 ? 'observed-cost' : 'observed-no-cost';
                    gridHTML += `<div class="grid-cell ${costClass}" id="${cellId}"></div>`;
                } else {
                    gridHTML += `<div class="grid-cell" id="${cellId}"></div>`;
                }
            }
        }
        gridHTML += `</div>`;
        return gridHTML;
    }
    
    // Record observed costs for a path
    recordObservedCosts(path, binaryCosts) {
        path.forEach(cell => {
            const [row, col] = cell;
            
            // Check for out-of-bounds error
            if (row < 0 || row > this.gridSize - 1 || col < 0 || col > this.gridSize - 1) {
                console.error(`Error in observed costs: Cell (${row}, ${col}) is out of bounds.`);
                return;
            }
    
            const cost = binaryCosts[row][col];
            this.observedCosts[`${row}-${col}`] = cost;

            // trialwise observed costs
            // const t = Math.floor(currentTrialIndex / this.nTrials);
            const t = currentTrialIndex - Math.floor(currentTrialIndex / grid.nTrials) * grid.nTrials;
            if (this[`observedCosts${t}`]) {
                this[`observedCosts${t}`][`${row}-${col}`] = cost;
            } else {
                console.error(`Error: observedCosts${t} is not defined.`);
            }

            // these costs are also available on all subsequent trials up to nTrials
            for (let i = t + 1; i < this.nTrials; i++) {
                if (this[`observedCosts${i}`]) {
                    this[`observedCosts${i}`][`${row}-${col}`] = cost;
                } else {
                    console.error(`Error: observedCosts${i} is not defined.`);
                }
            }
        });
    }    

    // Reset the grid for a new set of trials
    resetGrid() {
        this.observedCosts = {}; 
        this.currentGrid++; 
        console.log('currentGrid:', this.currentGrid);

        // Reset observed costs for each grid
        for (let i = 0; i < this.nTrials; i++) {
            this[`observedCosts${i}`] = {};
        }
    
        // Reset trial cost
        const trialCostElement = document.getElementById("trial-cost");
        if (trialCostElement) {
            trialCostElement.textContent = "-$0";
            trialCostElement.classList.add("hidden");
        }
    
        // Reset total cost
        totalCost = 0;
        const totalCostElement = document.getElementById("total-cost");
        if (totalCostElement) {
            totalCostElement.textContent = "$0";
        }
    }
    
    // Check if the city has changed for the upcoming trial
    hasCityChanged(trialIndex) {
        const upcomingTrial = this.getTrialInfo(trialIndex);
        const upcomingCity = upcomingTrial.city;
        
        // If this is the first trial, set the current city and return false
        if (this.currentCity === null) {
            this.currentCity = upcomingCity;
            return false;
        }
        
        // Check if the city has changed
        if (upcomingCity !== this.currentCity) {
            this.currentCity = upcomingCity;
            return true;
        }
        
        return false;
    }
    
    // Get the current city
    getCurrentCity() {
        return this.currentCity;
    }

    // Add createUpcomingJobsHTML as a method of the Grid class
    createUpcomingJobsHTML(currentTrialIndex) {
        const currentGridNumber = Math.floor(currentTrialIndex / this.nTrials);
        const currentGridStartIndex = currentGridNumber * this.nTrials;
        const currentGridEndIndex = currentGridStartIndex + this.nTrials - 1;

        const remainingTrialsInGrid = currentGridEndIndex - currentTrialIndex;
        if (remainingTrialsInGrid <= 0) {
            return '';
        }

        let upcomingHTML = `
            <div class="jobs-section">
                <div class="upcoming-jobs-header-container">
                    <div class="upcoming-jobs-header">Upcoming jobs</div>
                </div>
                <div class="upcoming-jobs-mask-container">
                    <div class="upcoming-jobs-actual-container">
        `;

        for (let i = 1; i <= remainingTrialsInGrid; i++) {
            const previewIndex = currentTrialIndex + i;
            const trial = this.getTrialInfo(previewIndex);

            upcomingHTML += `
                <div class="upcoming-job">
                    <div class="upcoming-grid" style="grid-template-columns: repeat(${this.gridSize}, 30px); grid-auto-rows: 30px;">
            `;

            for (let row = 0; row < this.gridSize; row++) {
                for (let col = 0; col < this.gridSize; col++) {
                    const isStartA = row === trial.start_A[0] && col === trial.start_A[1];
                    const isStartB = row === trial.start_B[0] && col === trial.start_B[1];
                    const isGoalA = row === trial.goal_A[0] && col === trial.goal_A[1];
                    const isGoalB = row === trial.goal_B[0] && col === trial.goal_B[1];
                    const isPathA = trial.path_A.some(coord => coord[0] === row && coord[1] === col);
                    const isPathB = trial.path_B.some(coord => coord[0] === row && coord[1] === col);

                    const observedCost = this.observedCosts[`${row}-${col}`];
                    const observedClass = observedCost !== undefined ? 
                        (observedCost === -1 ? 'observed-cost' : 'observed-no-cost') : '';

                    const isOverlap = isPathA && isPathB;
                    let pathClass = '';
                    let content = '';
                    if (isOverlap) {
                        const randomChoice = Math.random() < 0.5;
                        pathClass = randomChoice ? 'blue-path' : 'green-path';
                        content = randomChoice ? '<span class="green-text">⚝</span>' : '<span class="blue-text">⚝</span>';
                    } else if (isPathA) {
                        pathClass = 'blue-path';
                        content = '⚝';
                    } else if (isPathB) {
                        pathClass = 'green-path';
                        content = '⚝';
                    }

                    if (isStartA) {
                        upcomingHTML += `<div class="upcoming-cell ${observedClass} blue-path" data-row="${row}" data-col="${col}">
                                            <img src="assets/people/blue_person.png" alt="Blue Start" width="23" height="23">
                                         </div>`;
                    } else if (isStartB) {
                        upcomingHTML += `<div class="upcoming-cell ${observedClass} green-path" data-row="${row}" data-col="${col}">
                                            <img src="assets/people/green_person.png" alt="Green Start" width="23" height="23">
                                         </div>`;
                    } else if (isGoalA || isGoalB || isPathA || isPathB || isOverlap) {
                        upcomingHTML += `<div class="upcoming-cell ${observedClass} ${pathClass}" data-row="${row}" data-col="${col}" style="font-size: 1.5rem;">
                                            ${isGoalA || isGoalB ? '🏠' : content}
                                         </div>`;
                    } else {
                        upcomingHTML += `<div class="upcoming-cell ${observedClass}" data-row="${row}" data-col="${col}"></div>`;
                    }
                }
            }

            upcomingHTML += `
                    </div>
                </div>
            `;
        }

        upcomingHTML += `
                    </div>
                </div>
            </div>
        `;
        return upcomingHTML;
    }


    // Add createUpcomingJobsHTML as a method of the Grid class
    createAllJobsHTML(currentTrialIndex, selectedPath=null, keyAssignment=null, feedback=false, firstDay=false) {
        const trial = this.getTrialInfo(currentTrialIndex);
        const currentGridNumber = Math.floor(currentTrialIndex / this.nTrials);
        const currentGridStartIndex = currentGridNumber * this.nTrials;
        const currentGridEndIndex = currentGridStartIndex + this.nTrials - 1;
        const clockCharacters = ['&#x00E6;', '&#x00DD;', '&#x0026;', '&#x263A;']; // Add more characters if needed

        const totalTrialsInGrid = currentGridEndIndex - currentGridStartIndex + 1;

        let upcomingHTML = `
            <div class="jobs-section">
        `;

        if (!firstDay) {
            if (!feedback) {
                upcomingHTML += `
                <div id="cost-message" class="cost-display-container">
                <h2 class="day-display">Day ${trial.grid}/${this.nGrids}</h2>
                <h2 class="cost-total">Total Tolls Paid Today:</h2>
                <p id="total-cost" class="cost-total">-$${totalCost}</p>
                <p id="trial-cost" class="cost-trial hidden">-$0</p> 
                </div>
                `;
            } else {
                if (trial.grid === this.nGrids) {
                    upcomingHTML += `
                    <div id="cost-message" class="cost-display-container">
                    <h2 class="day-display">Day ${trial.grid}/${this.nGrids} Complete</h2>
                    <h2 class="cost-total">You paid a total of <strong style="color:  #f87171;">$${totalCost}</strong> today.</h2>
                    <p id="total-cost" class="cost-total">Press spacebar to continue.</p>
                    <p id="trial-cost" class="cost-trial hidden">-$0</p> 
                    </div>
                    `;
                } else {
                    upcomingHTML += `
                    <div id="cost-message" class="cost-display-container">
                    <h2 class="day-display">Day ${trial.grid}/${this.nGrids} Complete</h2>
                    <h2 class="cost-total">You paid a total of <strong style="color:  #f87171;">$${totalCost}</strong> today.</h2>
                    <p id="total-cost" class="cost-total">Tolls will now reset for the next day in this city. Press spacebar to continue.</p>
                    <p id="trial-cost" class="cost-trial hidden">-$0</p> 
                    </div>
                    `;
                }
            }
        } else if (firstDay) {
            upcomingHTML += `
            <div id="cost-message" class="cost-display-container">
            <h2 class="day-display">Day ${trial.grid}/${this.nGrids}</h2>
            <h2 class="cost-total">Here are your dispatches for the day.</h2>
            <p id="total-cost" class="cost-total">Get ready to select your jobs!</p>
            <p id="trial-cost" class="cost-trial hidden">-$0</p> 
            </div>
            `;
        }

        upcomingHTML += `
            <div class="upcoming-jobs-mask-container">
            <div class="upcoming-jobs-actual-container">
        `;
        

        for (let i = 0; i < totalTrialsInGrid; i++) {
            const previewIndex = currentGridStartIndex + i;
            const trial = this.getTrialInfo(previewIndex);
            const clockCharacter = clockCharacters[i]; // Cycle through the characters based on the trial index
            
            if (!firstDay) {
                if (!feedback) {
                    if (previewIndex < currentTrialIndex) {
                        upcomingHTML += `
                            <div class="upcoming-job">
                                <div class="clock-container" style="font-size: 50px; text-align: center; margin-bottom: 10px; color: transparent;">
                                    ${clockCharacter}
                                </div>
                                <div class="upcoming-grid-done" style="grid-template-columns: repeat(${this.gridSize}, 30px); grid-auto-rows: 30px;">
                        `;
                    } else if (previewIndex === currentTrialIndex) {
                        upcomingHTML += `
                            <div class="upcoming-job">
                                <div class="clock-container" style="font-size: 50px; text-align: center; margin-bottom: 10px; color: #ece75d;">
                                    ${clockCharacter}
                                </div>
                                <div class="upcoming-grid" style="grid-template-columns: repeat(${this.gridSize}, 30px); grid-auto-rows: 30px; background-color: ${keyAssignment ? '#ece75d' : ''};">
                        `;
                    } else {
                        upcomingHTML += `
                            <div class="upcoming-job">  
                                <div class="clock-container" style="font-size: 50px; text-align: center; margin-bottom: 10px; color: ${previewIndex === currentTrialIndex ? '#ece75d' : 'inherit'};">
                                    ${clockCharacter}
                                </div>
                                <div class="upcoming-grid" style="grid-template-columns: repeat(${this.gridSize}, 30px); grid-auto-rows: 30px;">
                        `;
                    }
                } else if (feedback) {
                    upcomingHTML += `
                            <div class="upcoming-job">
                                <div class="clock-container" style="font-size: 50px; text-align: center; margin-bottom: 10px; color: transparent;">
                                    ${clockCharacter}
                                </div>
                                <div class="upcoming-grid-done" style="grid-template-columns: repeat(${this.gridSize}, 30px); grid-auto-rows: 30px;">
                    `;
                }
            } else if (firstDay) {
                upcomingHTML += `
                            <div class="upcoming-job">  
                                <div class="clock-container" style="font-size: 50px; text-align: center; margin-bottom: 10px;">
                                    ${clockCharacter}
                                </div>
                                <div class="upcoming-grid" style="grid-template-columns: repeat(${this.gridSize}, 30px); grid-auto-rows: 30px;">
                        `;
            }

            for (let row = 0; row < this.gridSize; row++) {
                for (let col = 0; col < this.gridSize; col++) {
                    const cellId = `cell-${row}-${col}-trial-${trial.trial}`;
                    let isStartA, isStartB, isGoalA, isGoalB, isPathA, isPathB;

                    // if (previewIndex !== currentTrialIndex) {
                    if (previewIndex > currentTrialIndex) {
                        isStartA = row === trial.start_A[0] && col === trial.start_A[1];
                        isStartB = row === trial.start_B[0] && col === trial.start_B[1];
                        isGoalA = row === trial.goal_A[0] && col === trial.goal_A[1];
                        isGoalB = row === trial.goal_B[0] && col === trial.goal_B[1];
                        isPathA = trial.path_A.some(coord => coord[0] === row && coord[1] === col);
                        isPathB = trial.path_B.some(coord => coord[0] === row && coord[1] === col);
                    } else if (previewIndex < currentTrialIndex) {
                            isStartA = false;
                            isStartB = false;
                            isGoalA = false;
                            isGoalB = false;
                            isPathA = false;
                            isPathB = false;
                    } else {
                        isStartA = selectedPath !== 'green' && selectedPath !== 'none' && row === trial.start_A[0] && col === trial.start_A[1];
                        isStartB = selectedPath !== 'blue' && selectedPath !== 'none' && row === trial.start_B[0] && col === trial.start_B[1];
                        isGoalA = selectedPath !== 'green' && selectedPath !== 'none' && row === trial.goal_A[0] && col === trial.goal_A[1];
                        isGoalB = selectedPath !== 'blue' && selectedPath !== 'none' && row === trial.goal_B[0] && col === trial.goal_B[1];
                        isPathA = selectedPath !== 'green' && selectedPath !== 'none' && trial.path_A.some(coord => coord[0] === row && coord[1] === col);
                        isPathB = selectedPath !== 'blue' && selectedPath !== 'none' && trial.path_B.some(coord => coord[0] === row && coord[1] === col);
                    } 
                    
                    // plot observed costs in every grid
                    // const observedCost = this.observedCosts[`${row}-${col}`];
                    // const observedClass = observedCost !== undefined ? 
                    // (observedCost === -1 ? 'observed-cost' : 'observed-no-cost') : '';

                    // or, only plot costs observed up to the current trial, i.e. use this.observedCostsT to get the observed costs
                    const observedCost = this[`observedCosts${i}`][`${row}-${col}`];
                    const observedClass = observedCost !== undefined ?
                        (observedCost === -1 ? 'observed-cost' : 'observed-no-cost') : '';
                        
                    // Handle overlapping paths
                    const isOverlap = isPathA && isPathB;
                    let pathClass = '';
                    let content = '';

                    // Use key assignment if previewIndex matches currentTrialIndex
                    if (previewIndex === currentTrialIndex && keyAssignment) {
                        if (isOverlap) {
                            if (previewIndex % 2 === 0) {
                                pathClass = 'blue-path';
                                content = `<span class="green-text">⚝</span>`;
                            } else {
                                pathClass = 'green-path';
                                content = `<span class="blue-text">⚝</span>`;
                            }
                        } else if (isPathA) {
                            pathClass = 'blue-path';
                            content = keyAssignment.blue;
                        } else if (isPathB) {
                            pathClass = 'green-path';
                            content = keyAssignment.green;
                        }
                    } else {
                        // Default behavior for other previews
                        if (isOverlap) {
                            if (i % 2 === 0) {
                                pathClass = 'blue-path';
                                content = '<span class="green-text">⚝</span>';
                            } else {
                                pathClass = 'green-path';
                                content = '<span class="blue-text">⚝</span>';
                            }
                        } else if (isPathA) {
                            pathClass = 'blue-path';
                            content = '⚝';
                        } else if (isPathB) {
                            pathClass = 'green-path';
                            content = '⚝';
                        }
                    }

                    if (previewIndex < currentTrialIndex || feedback) {
                        if (isStartA) {
                            upcomingHTML += `<div class="upcoming-cell-done ${observedClass} blue-path" id="${cellId}" data-row="${row}" data-col="${col}">
                                                <img src="assets/people/blue_person.png" alt="Blue Start" width="23" height="23">
                                             </div>`;
                        } else if (isStartB) {
                            upcomingHTML += `<div class="upcoming-cell-done ${observedClass} green-path" id="${cellId}" data-row="${row}" data-col="${col}">
                                                <img src="assets/people/green_person.png" alt="Green Start" width="23" height="23">
                                             </div>`;
                        } else if (isGoalA || isGoalB || isPathA || isPathB || isOverlap) {
                            upcomingHTML += `<div class="upcoming-cell-done ${observedClass} ${pathClass}" id="${cellId}" data-row="${row}" data-col="${col}" style="font-size: 1.6rem;">
                                                ${isGoalA || isGoalB ? '🏠' : content}
                                             </div>`;
                        } else {
                            upcomingHTML += `<div class="upcoming-cell-done ${observedClass}" id="${cellId}" data-row="${row}" data-col="${col}"></div>`;
                        }
                    } else {
                        if (isStartA) {
                            upcomingHTML += `<div class="upcoming-cell ${observedClass} blue-path" id="${cellId}" data-row="${row}" data-col="${col}">
                                                <img src="assets/people/blue_person.png" alt="Blue Start" width="23" height="23">
                                             </div>`;
                        } else if (isStartB) {
                            upcomingHTML += `<div class="upcoming-cell ${observedClass} green-path" id="${cellId}" data-row="${row}" data-col="${col}">
                                                <img src="assets/people/green_person.png" alt="Green Start" width="23" height="23">
                                             </div>`;
                        } else if (isGoalA || isGoalB || isPathA || isPathB || isOverlap) {
                            upcomingHTML += `<div class="upcoming-cell ${observedClass} ${pathClass}" id="${cellId}" data-row="${row}" data-col="${col}" style="font-size: 1.6rem;">
                                                ${isGoalA || isGoalB ? '🏠' : content}
                                             </div>`;
                        } else {
                            upcomingHTML += `<div class="upcoming-cell ${observedClass}" id="${cellId}" data-row="${row}" data-col="${col}"></div>`;
                        }
                    }
                }
            }
            

            upcomingHTML += `
                    </div>
                </div>
            `;
        }

        upcomingHTML += `
                    </div>
                </div>
            </div>
        `;

        return upcomingHTML;
    }
}

// rename sequence to data, and then use this to generate the grid
// let grid;
// let data;
// data = sequence;
// let currentTrialIndex = 0;
// grid = new Grid(data); // Initialize the Grid class with the loaded data
// const numCities = data.env_costs.n_cities; // Assuming this is the number of cities
// createCityMapping(numCities);
// console.log('Grid data loaded:', grid);
// console.log('City mapping created:', cityMapping);

// Function to load practice grid data
function loadPracticeGrid(filePath, gridVariableName) {
    return fetch(filePath)
    .then(response => response.json())
    .then(data => {
        const practiceGrid = new Grid(data); // Initialize the Grid class with the loaded data
        console.log(`${gridVariableName} data loaded:`, practiceGrid);
        return practiceGrid;
    })
    .catch(error => console.error(`Error loading ${gridVariableName} JSON:`, error));
}

// Load practice grids
let practice1Grid, practice2Grid, practice3Grid, practice4Grid;
let practice1TrialIndex = 0, practice2TrialIndex = 0, practice3TrialIndex = 0, practice4TrialIndex = 0;

Promise.all([
    loadPracticeGrid('assets/trial_sequences/practice_sequence1.json', 'practice1Grid').then(grid => practice1Grid = grid),
    loadPracticeGrid('assets/trial_sequences/practice_sequence2.json', 'practice2Grid').then(grid => practice2Grid = grid),
    loadPracticeGrid('assets/trial_sequences/practice_sequence3.json', 'practice3Grid').then(grid => practice3Grid = grid),
    loadPracticeGrid('assets/trial_sequences/practice_sequence4.json', 'practice4Grid').then(grid => practice4Grid = grid)
]).then(() => {
    console.log('Both practice grids loaded successfully.');
}).catch(error => {
    console.error('Error loading practice grids:', error);
});


// Function to create a random mapping of city IDs
let cityMapping = {}; // This will store our shuffled mapping
function createCityMapping(numCities) {
    // Create an array of city IDs (1 to numCities)
    let cityIds = Array.from({length: numCities}, (_, i) => i + 1);
    
    // Shuffle the array
    cityIds = shuffleArray(cityIds);
    
    // Create the mapping
    for (let i = 1; i <= numCities; i++) {
        cityMapping[i] = cityIds[i-1];
    }
}

// Fisher-Yates shuffle algorithm
function shuffleArray(array) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
    return array;
}

// Function to animate the agent along the chosen path
let totalCost = 0; // Keeps track of total cost across trials

// 1. Add taxi character with animation
function createAvatar() {
    return `
        🚖
    `;
}

// 2. Add visual and audio feedback for costs
function animateAgent(path, binaryCosts, callback) {
    let currentStep = 0;
    let trialCost = 0;
    let trialCostVisible = false;

    if (path) {
        function step() {
            if (currentStep > 0) {
                const [prevRow, prevCol] = path[currentStep - 1];
                const trial = grid.getTrialInfo(currentTrialIndex);
                const prevCellElement = document.getElementById(`cell-${prevRow}-${prevCol}-trial-${trial.trial}`);

                if (prevCellElement) {
                    prevCellElement.classList.remove('avatar');
                    prevCellElement.innerHTML = ''; // Remove avatar
                }
            }

            if (currentStep < path.length) {
                const [curRow, curCol] = path[currentStep];
                const trial = grid.getTrialInfo(currentTrialIndex);
                const cellElement = document.getElementById(`cell-${curRow}-${curCol}-trial-${trial.trial}`);

                if (cellElement) {
                    const cost = binaryCosts[curRow][curCol];

                    // Update observed cost classes
                    cellElement.classList.remove("observed-cost", "observed-no-cost");
                    cellElement.classList.add(cost === -1 ? "observed-cost" : "observed-no-cost");

                    // Ensure start and goal cells update their color when observed
                    if (cellElement.classList.contains("start") || cellElement.classList.contains("goal")) {
                        cellElement.style.backgroundColor = cost === -1 ? "#f87171" : "#b8b8d9"; // Red for toll, grey for free
                    }

                    if (cost === -1) {
                        trialCost++;

                        // Visual burst effect for toll cost
                        cellElement.innerHTML += '<div class="cost-burst">+$1 Toll</div>';
                        setTimeout(() => {
                            const burst = cellElement.querySelector('.cost-burst');
                            if (burst) burst.remove();
                        }, 600);

                        // Play toll sound
                        // costSound.play();
                        playCostSound();

                        if (!trialCostVisible) {
                            const trialCostElement = document.getElementById("trial-cost");
                            if (trialCostElement) {
                                trialCostElement.classList.remove("hidden");
                                trialCostVisible = true;
                            }
                        }
                    } else {
                        // Visual feedback for free passage
                        cellElement.innerHTML += '<div class="free-burst">Free</div>';
                        setTimeout(() => {
                            const burst = cellElement.querySelector('.free-burst');
                            if (burst) burst.remove();
                        }, 600);
                    }

                    const trialCostElement = document.getElementById("trial-cost");
                    if (trialCostElement) {
                        trialCostElement.textContent = `-$${trialCost}`;
                    }

                    // Update observed costs in upcoming grids
                    const upcomingCells = document.querySelectorAll(`.upcoming-cell[data-row="${curRow}"][data-col="${curCol}"]`);
                    upcomingCells.forEach(upcomingCell => {
                        upcomingCell.classList.remove("observed-cost", "observed-no-cost");
                        upcomingCell.classList.add(cost === -1 ? "observed-cost" : "observed-no-cost");
                    });

                    // Remove the star or S or G, then add the avatar
                    cellElement.textContent = '';
                    cellElement.classList.add('avatar');
                    cellElement.innerHTML += createAvatar(); // Add taxi avatar on top
                } else {
                    console.error(`Cell not found in DOM: cell-${curRow}-${curCol}`);
                    return;
                }

                currentStep++;
                setTimeout(step, 300);
            } else {
                // Animation complete
                mergeCosts(trialCost, callback);
            }
        }
        
        // Start the animation sequence after a short delay
        setTimeout(step, 600);
    } else {
        // If there's no path, just merge costs and execute callback
        mergeCosts(null, callback);
    }
    // Remove this line: setTimeout(step, 600);
}

// 4. Add animated transitions between trials
// 4. Add animated transitions between trials
function mergeCosts(trialCost, callback) {
    const totalCostElement = document.getElementById("total-cost");
    const trialCostElement = document.getElementById("trial-cost");

    let trialFine;
    if (trialCost === null) {
        trialFine = true;
        trialCost = 10;
        
        // Show missed response message in red for n seconds
        if (totalCostElement) {
            const originalContent = totalCostElement.textContent;
            const originalColor = totalCostElement.style.color;
            
            totalCostElement.textContent = `You ran out of time! -$${trialCost}`;
            totalCostElement.style.color = "#f87171";
            
            // Wait n seconds before continuing
            setTimeout(() => {
                totalCostElement.style.color = originalColor;
                
                // Continue with normal animation flow after showing the message
                if (totalCostElement && trialCostElement) {
                    // Add warning animation to cost display
                    if (trialCost > 0) {
                        trialCostElement.classList.add("cost-animate");
                    }
                    
                    trialCostElement.style.transition = "transform 0.5s ease-in-out";
                    trialCostElement.style.transform = "translateY(-20px)";

                    setTimeout(() => {
                        totalCost += trialCost;
                        
                        // Animated counter for total cost
                        const startCost = totalCost - trialCost;
                        const duration = 500;
                        const frameDuration = 1000 / 60;
                        const totalFrames = Math.round(duration / frameDuration);
                        let frame = 0;
                        
                        const counter = setInterval(() => {
                            frame++;
                            const progress = frame / totalFrames;
                            const currentCount = Math.floor(startCost + progress * trialCost);
                            totalCostElement.textContent = `$${currentCount}`;
                            
                            if (frame === totalFrames) {
                                clearInterval(counter);
                                totalCostElement.textContent = `-$${totalCost}`;

                                // Reset trial cost display with animation
                                trialCostElement.textContent = `-$0`;
                                trialCostElement.classList.remove("cost-animate");
                                trialCostElement.style.transform = "translateY(0)";
                                trialCostElement.classList.add("hidden");
                            }
                        }, frameDuration);
                    }, 100);
                }
            }, 1000);
        }
    } else {
        trialFine = false;
        
        // Regular flow without missed response
        if (totalCostElement && trialCostElement) {
            // Add warning animation to cost display
            if (trialCost > 0) {
                trialCostElement.classList.add("cost-animate");
            }
            
            trialCostElement.style.transition = "transform 0.5s ease-in-out";
            trialCostElement.style.transform = "translateY(-20px)";

            setTimeout(() => {
                totalCost += trialCost;
                
                // Animated counter for total cost
                const startCost = totalCost - trialCost;
                const duration = 500;
                const frameDuration = 1000 / 60;
                const totalFrames = Math.round(duration / frameDuration);
                let frame = 0;
                
                const counter = setInterval(() => {
                    frame++;
                    const progress = frame / totalFrames;
                    const currentCount = Math.floor(startCost + progress * trialCost);
                    totalCostElement.textContent = `$${currentCount}`;
                    
                    if (frame === totalFrames) {
                        clearInterval(counter);
                        totalCostElement.textContent = `$${totalCost}`;

                        // Reset trial cost display with animation
                        trialCostElement.textContent = `-$0`;
                        trialCostElement.classList.remove("cost-animate");
                        trialCostElement.style.transform = "translateY(0)";
                        trialCostElement.classList.add("hidden");
                    }
                }, frameDuration);
            }, 100);
        }
    }

    setTimeout(() => {
        const currentJob = document.querySelector(".grid-container");
        const upcomingJobs = document.querySelectorAll(".upcoming-job");
        if (upcomingJobs.length > 0) {
            
            // Add transition effect to fade out the current job
            // if (currentJob) {
            //     currentJob.classList.add("fade-out");
            // }

            // define currentJob as the upcoming job indexed by the current trial
            const currentI = currentTrialIndex - Math.floor(currentTrialIndex / grid.nTrials) * grid.nTrials;
            const currentJob = upcomingJobs[currentI];
            // currentJob.classList.add("transparent");
            currentJob.classList.add("fade-out");


            // Add transition effect to fade out the leftmost upcoming job
            // const leftmostJob = upcomingJobs[0];
            // if (leftmostJob) {
            //     leftmostJob.classList.add("fade-out");
            // }

            setTimeout(() => {
                currentTrialIndex++;
                jsPsych.finishTrial();
                setTimeout(() => {
                    // Remove fade-transition class after the transition
                    if (currentJob) currentJob.classList.remove("fade-out");
                    // if (currentJob) currentJob.classList.remove("transparent");
                    // if (leftmostJob) leftmostJob.classList.remove("fade-out");
                }, 500);
            }, 500);
        } else {
            setTimeout(() => {
                currentTrialIndex++;
                jsPsych.finishTrial();
            }, 500);
        }
    }, 1000);
}


const firstDayTrial = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {
        return `
            <div class="jobs-layout">
            <div class="upcoming-jobs-container grid ${currentTrialIndex % grid.nTrials === 0 ? 'grid-fade-in' : ''}">
                ${grid.createAllJobsHTML(currentTrialIndex, null, null, false, true)} 
            </div>
            </div>
        `;
    },
    choices: "NO_KEYS",
    trial_duration: 3000, // 
    on_finish: function() {
    }
};

const pathPreSelectionTrial = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {
        return `
            <div class="jobs-layout">
            <div class="upcoming-jobs-container grid">
                ${grid.createAllJobsHTML(currentTrialIndex, null, null)} 
            </div>
            </div>
        `;
    },
    choices: "NO_KEYS",
    trial_duration: 3000, 
    on_finish: function() {
    }
};

// 5. Update the path selection trial to include taxi theme elements
const pathSelectionTrial = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {
        // Randomly assign F and J to blue and green paths
        const keyAssignment = Math.random() < 0.5 ? 
            { blue: 'F', green: 'J' } : 
            { blue: 'J', green: 'F' };
        
        // Store the assignment for this trial
        jsPsych.data.addProperties({
            blue_key: keyAssignment.blue,
            green_key: keyAssignment.green
        });
        
        return `
            <div class="jobs-layout">
                <div class="upcoming-jobs-container">
                    ${grid.createAllJobsHTML(currentTrialIndex, null, keyAssignment)} 
                </div>
            </div>
        `;
    },
    choices: ['f', 'j'], 
    trial_duration: 7000, // Automatically ends after 5 seconds
    on_finish: function(data) {
        // Get the key assignment for this trial
        const keyAssignment = {
            blue: jsPsych.data.get().last(1).values()[0].blue_key,
            green: jsPsych.data.get().last(1).values()[0].green_key
        };
        
        // Determine the chosen path based on the key pressed
        let choice;
        if (data.response === 'f') {
            choice = keyAssignment.blue === 'F' ? 'blue' : 'green';
        } else if (data.response === 'j') {
            choice = keyAssignment.blue === 'J' ? 'blue' : 'green';
        } else {
            choice = 'nan'; // Log as 'nan' if no response is made
        }
        
        // Add "swipe" effect on selection
        const choiceElement = document.getElementById(`${choice}-choice`);
        const unchosenElement = document.getElementById(choice === 'blue' ? 'green-choice' : 'blue-choice');
        
        if (choiceElement && unchosenElement) {
            choiceElement.classList.add('choice-selected');
            unchosenElement.classList.add('choice-unselected');
        }
    
        // Replot the grid with only the chosen path
        const gridContainer = document.querySelector(".current-job-section");
        if (gridContainer) {
            gridContainer.innerHTML = grid.createGridHTML(currentTrialIndex, choice, keyAssignment);
        }
        
        // Store all the relevant data from the current trial
        const currentTrial = grid.getTrialInfo(currentTrialIndex);
        data.choice = choice;
        data.trial = currentTrial.trial;
        data.city = currentTrial.city;
        data.grid_id = currentTrial.grid;
        data.cityID = cityMapping[currentTrial.city];
        data.path_chosen = choice;
        data.button_pressed = data.response;
        data.reaction_time_ms = data.rt;
        data.key_assignment = keyAssignment;
        data.path_A_expected_cost = currentTrial.path_A_expected_cost;
        data.path_B_expected_cost = currentTrial.path_B_expected_cost;
        data.path_A_actual_cost = currentTrial.path_A_actual_cost;
        data.path_B_actual_cost = currentTrial.path_B_actual_cost;
        data.path_A_future_overlap = currentTrial.path_A_future_overlap;
        data.path_B_future_overlap = currentTrial.path_B_future_overlap;
        data.dominant_axis_A = currentTrial.dominant_axis_A;
        data.dominant_axis_B = currentTrial.dominant_axis_B;
        data.abstract_sequence_A = JSON.stringify(currentTrial.abstract_sequence_A);
        data.abstract_sequence_B = JSON.stringify(currentTrial.abstract_sequence_B);
        data.better_path = currentTrial.better_path;
        const better_path_ID = currentTrial.better_path === 'a' ? 'blue' : currentTrial.better_path === 'b' ? 'green' : null;
        if (choice === better_path_ID) {
            data.chose_better_path = 1;
        } else {
            data.chose_better_path = 0;
        }

        // Include all columns from the current trial
        Object.keys(currentTrial).forEach(key => {
            data[key] = currentTrial[key];
        });

        // Include all trial info from the current trial
        Object.assign(data, currentTrial);
        
        // Add the trial data to jsPsych's data collection
        jsPsych.data.get().addToLast(data);
    }
};

const pathAnimationTrial = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {
        const lastTrialData = jsPsych.data.get().last(1).values()[0];
        const keyAssignment = lastTrialData.key_assignment;
        return `
            <div class="jobs-layout">
                <div class="upcoming-jobs-container">
                    ${grid.createAllJobsHTML(currentTrialIndex, lastTrialData.choice, keyAssignment)} 
                </div>
            </div>
        `;
    },
    choices: "NO_KEYS",
    on_load: function() {
        const currentTrial = grid.getTrialInfo(currentTrialIndex);
        const lastTrialData = jsPsych.data.get().last(1).values()[0];

        if (!lastTrialData || !lastTrialData.choice) {
            console.error("No valid path choice found. Restarting trial.");
            return jsPsych.finishTrial();
        }

        const chosenPath = lastTrialData.choice === 'blue' ? currentTrial.path_A : 
                   lastTrialData.choice === 'green' ? currentTrial.path_B : 
                   null;
        const binaryCosts = grid.getBinaryCosts(`city_${currentTrial.city}_grid_${currentTrial.grid}`);

        if (chosenPath !== null) {
            grid.recordObservedCosts(chosenPath, binaryCosts);
        }

        setTimeout(() => {
            animateAgent(chosenPath, binaryCosts, function() {
                jsPsych.finishTrial();
            });
        }, 100);
    }
};


// Update the setCityBackground function to use the mapping
function setCityBackground(cityId) {
    const body = document.body;
    
    // Use the mapped city ID if cityId is not 'practice1' or 'practice2'
    const mappedCityId = (cityId === 'practice1' || cityId === 'practice2' || cityId === 'practice3') ? cityId : cityMapping[cityId];

    // Clear any existing background styles before applying a new one
    body.style.backgroundImage = '';
    body.style.backgroundSize = '';
    body.style.backgroundPosition = '';
    body.style.backgroundRepeat = '';

    // Apply the new background
    body.style.backgroundImage = `url('assets/cities/cropped/${mappedCityId}.png')`;
    body.style.backgroundSize = 'cover';
    body.style.backgroundPosition = 'center';
    body.style.backgroundRepeat = 'no-repeat';
}

// Modified function structure: separate functions for city change vs day change
function animateCityChange(oldCityId, newCityId) {
    
    // Create a container for the animation
    let transitionContainer = document.createElement('div');
    transitionContainer.style.position = 'fixed';
    transitionContainer.style.top = '0';
    transitionContainer.style.left = '0';
    transitionContainer.style.width = '200%'; // Double width to fit both images
    transitionContainer.style.height = '100%';
    transitionContainer.style.zIndex = '1000';
    transitionContainer.style.display = 'flex';
    transitionContainer.style.transition = 'transform 1.5s ease-in-out';
    document.body.appendChild(transitionContainer);
    
    // Create old city element
    let oldCity = document.createElement('div');
    let oldCityMapping;
    oldCity.style.width = '50%'; // Half of the container
    oldCity.style.height = '100%';
    oldCityMapping = cityMapping[oldCityId];
    oldCity.style.backgroundImage = `url('assets/cities/cropped/${oldCityMapping}.png')`;
    oldCity.style.backgroundSize = 'cover';
    oldCity.style.backgroundPosition = 'center';
    transitionContainer.appendChild(oldCity);
    
    // Create new city element
    let newCity = document.createElement('div');
    let newCityMapping;
    newCity.style.width = '50%'; // Half of the container
    newCity.style.height = '100%';
    newCityMapping = cityMapping[newCityId];
    newCity.style.backgroundImage = `url('assets/cities/cropped/${newCityMapping}.png')`;
    newCity.style.backgroundSize = 'cover';
    newCity.style.backgroundPosition = 'center';
    transitionContainer.appendChild(newCity);
    
    // Force browser reflow before starting animation
    void transitionContainer.offsetWidth;
    
    // Start the slide animation
    transitionContainer.style.transform = 'translateX(-50%)';
    
    // After animation completes, set the new background and remove the container
    setTimeout(() => {
        setCityBackground(newCityId);
        document.body.removeChild(transitionContainer);
    }, 1600);
}

function animateDayChange(cityId) {

    const blackCover = document.createElement('div');
    blackCover.style.position = 'fixed';
    blackCover.style.top = '0';
    blackCover.style.left = '0';
    blackCover.style.width = '100%';
    blackCover.style.height = '100%';
    blackCover.style.backgroundColor = 'black';
    blackCover.style.opacity = '0';
    blackCover.style.transition = 'opacity 1s ease-in-out';
    blackCover.style.zIndex = '1000';
    document.body.appendChild(blackCover);

    // Fade to full opacity
    setTimeout(() => {
        blackCover.style.opacity = '0.5';
    }, 10);

    // After 1s, set the new city background and fade back to transparency
    setTimeout(() => {
        setCityBackground(cityId);
        blackCover.style.opacity = '0';
    }, 1000);

    // Remove the black cover and grid after the transition is complete
    setTimeout(() => {
        document.body.removeChild(blackCover);
    }, 2000);
}

// display feedback after a grid
const gridFeedback = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {
        const todayTolls = totalCost; // Assuming totalCost tracks the tolls paid so far
        return `
            <div class="new-day-text">
                <h3>You paid a total of <strong style="color:  #f87171;">$${todayTolls}</strong> in tolls today.</h3>
                <h3>Press spacebar to continue.</h3>
            </div>
        `;
    },
choices: [' '], // spacebar to continue
    // on_finish: function() {
    //     grid.resetGrid(); // Reset the grid for the next set of trials
    // }
};

// display feedback after a grid
const practiceGridFeedback = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {
        const todayTolls = totalCost; // Assuming totalCost tracks the tolls paid so far
        return `
            <div class="new-day-text">
                <h3>You would have paid a total of <strong style="color:  #f87171;">$${todayTolls}</strong> in tolls today.</h3>
                <h3>Press spacebar to continue.</h3>
            </div>
        `;
    },
choices: [' '], // spacebar to continue
};

const timeoutCheck = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {
        // Get the current city ID and the previous city ID
        const currentCityId = grid.getCurrentCity();
        const previousCityId = currentCityId - 1;

        // Retrieve all trials from the previous city
        const previousCityTrials = jsPsych.data.get().filterCustom(function(trial) {
            return trial.city === previousCityId;
        });

        // Count the number of timeouts (where choice is 'nan')
        const timeouts = previousCityTrials.filter(trial => trial.choice === 'nan').count();
        console.log(`Number of timeouts in city ${previousCityId}:`, timeouts);

        // Check if the number of timeouts exceeds the threshold
        const nTrials = grid.nTrials;
        const nGrids = grid.nGrids;
        const nTrialsPerCity = nTrials * nGrids;
        const threshold = Math.floor(0.3 * nTrialsPerCity);
        console.log(`Threshold for timeouts: ${threshold}`);

        if (timeouts > threshold) {
            console.log(`Participant failed due to high number of timeouts in city ${previousCityId}`);
            setTimeout(() => {
                // window.location.href = "YOUR_REDIRECT_URL_HERE"; // Replace with your URL
                window.location.replace("error.html");
            }, 5000); // Redirect after 5 seconds
            return `
                <div class="error-message">
                    <h2>Experiment Failed</h2>
                    <p>You have timed out too many times in the previous city. Unfortunately, you cannot continue with the experiment.</p>
                    <p>You will now be redirected to Prolific.</p>
                </div>
            `;
        } else {
            // If successful, just end the trial
            jsPsych.finishTrial();
            return null;
        }
    },
    choices: "NO_KEYS", // Disable keypresses
};

const newCityMessage = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {
        const nextTrialIndex = currentTrialIndex; // Next trial will be this index
        const nextTrial = grid.getTrialInfo(nextTrialIndex);
        const nCities = grid.nCities
        const nextCityId = nextTrial.city;
        const currentCityId = grid.getCurrentCity();
        
        let message;
        console.log("City changed from", currentCityId, "to", nextCityId);
        animateCityChange(currentCityId, nextCityId);
        
        message = `
        <div class="new-day-text">
            <div>
                <h2>City ${currentCityId-1}/${nCities} complete.</h2>
                <h2>New City!</h2>
                <p>Your taxi company is now operating in a new city.</p>
                <p>Note: this may (or may not) be a different type of city - i.e. the traffic either tends to run from north-south, or east-west.</p>
                <p>Prepare for your first day in this new city.</p>
                <p id="continue-text" style="display: none;">Press spacebar to continue dispatching.</p>
            </div>
        </div>
        `;
        
        // Update the current city
        grid.currentCity = nextCityId;

        // After 2s, show the text and enable keypresses manually
        setTimeout(() => {
        document.getElementById("continue-text").style.display = "block";
        
        // Manually register keypress listener
        jsPsych.pluginAPI.getKeyboardResponse({
            callback_function: jsPsych.finishTrial, // Ends trial when the spacebar is pressed
            valid_responses: [' '], // Only allow the spacebar
            rt_method: "performance",
            persist: false,
            allow_held_key: false
        });
        }, 2500); // Increased delay to allow animation to complete

        return message;
    },
    choices: "NO_KEYS", // Initially disable keypresses
    on_finish: function() {
        grid.resetGrid(); // Reset the grid for the new set of trials
        var ppt_data = jsPsych.data.get().json();
        send_incomplete(subject_id, ppt_data);
    }
};

const newDayMessage = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {
        const feedback = true;
        return `
            <div class="jobs-layout">
                <div class="upcoming-jobs-container grid">
                    ${grid.createAllJobsHTML(currentTrialIndex - 1, 'none', null, feedback)} 
                </div>
            </div>
        `;
    },
    choices: [' '], // Spacebar to continue
    on_finish: function() {
        grid.resetGrid(); // Reset the grid for the new set of trials
        var ppt_data = jsPsych.data.get().json();
        send_incomplete(subject_id, ppt_data);
    }
};

const firstGridMessage = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {

        // revert back to first trial
        currentTrialIndex = 0;
        totalCost = 0;

        // Set initial city background from the first trial
        const firstTrial = grid.getTrialInfo(0);
        const cityId = firstTrial.city;
        console.log('trialIndex:', currentTrialIndex);
        console.log('cityId:', cityId);
        grid.currentCity = cityId; // Initialize the current city
        setCityBackground(cityId); // Plot the first city background

        return `
            <div class="new-day-text">
                <h2>Ready?</h2>
                <p>Your taxi company is starting operations in its first city.</p>
                <p>Remember: your goal is to minimise the total tolls paid each day.</p>
                <p>Press spacebar to begin dispatching.</p>
            </div>
        `;
    },
choices: [' '], // spacebar to continue
    on_finish: function() {
        console.log("Experiment has begun in City:", grid.getCurrentCity()), ', Trial:', currentTrialIndex,', Grid:', grid.getTrialInfo(currentTrialIndex).grid;
        var ppt_data = jsPsych.data.get().json();
        send_incomplete(subject_id, ppt_data);
        // send_incomplete(subject_id, jsPsych.data);
    }
};

// End message
const end = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {
        return `
            <div class="new-day-text">
                <h1>Shift Complete!</h1>
                <p>Great job, Dispatcher!</p>
                <p>You've successfully completed all taxi assignments.</p>
                <p>Your performance data has been recorded for evaluation.</p>
                <p>Press spacebar to see if you received your bonus.</p>
            </div>
        `;
    },
choices: [' '], // spacebar to continue
    on_load: function() {
    }
};

// calculate bonus
let bonusAchieved;
const bonus = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {
        bonusAchieved = calculateBonus();
        console.log("Bonus Achieved:", bonusAchieved);
        const bonusMessage = bonusAchieved
            ? "Congratulations! You earned a bonus!"
            : "Unfortunately, you did not earn a bonus this time.";
        return `
            <div class="new-day-text">
                <h2>${bonusMessage}</h2>
                <p>Press spacebar to return to Prolific.</p>
            </div>
        `;
    },
choices: [' '], // spacebar to continue
    on_finish: function() {
        data.bonusAchieved = bonusAchieved;
    }
};

// Function to calculate bonus
function calculateBonus() {
    
    // Randomly select nGrids cities without replacement
    const cityIndices = Array.from({ length: grid.nCities }, (_, i) => i + 1); // Create an array of city indices (1 to nCities)
    const shuffledCities = shuffleArray(cityIndices); // Shuffle the array
    const selectedCities = shuffledCities.slice(0, grid.nGrids); // Take the first nGrids cities

    // Calculate the average accuracy for the selected grids
    let totalAccuracy = 0;
    selectedCities.forEach((cityIndex, gridIndex) => {
        console.log("City Index:", cityIndex, "Grid Index:", gridIndex);
        const trials = jsPsych.data.get().filterCustom(function(trial) {
            return trial.city === cityIndex && trial.grid === gridIndex+1;
        }).values();
        console.log("Trials for City:", cityIndex, "Grid:", gridIndex, trials);
        if (trials.length === 0) {
            console.warn(`No trials found for city ${cityIndex} in grid ${gridIndex}.`);
            return;
        }
        // sum the accuracy in these trials
        const accuracy = trials.reduce((sum, trial) => {
            return sum + (trial.chose_better_path ? 1 : 0);
        }, 0) / trials.length; // Average accuracy for this grid
        console.log("Accuracy for City:", cityIndex, "Grid:", gridIndex, accuracy);
        totalAccuracy += accuracy;
    });

    const averageAccuracy = totalAccuracy / grid.nGrids;

    // debugging: print everything
    console.log("Selected Cities:", selectedCities);
    console.log("Total Accuracy:", totalAccuracy);
    console.log("Average Accuracy:", averageAccuracy);
    console.log("Bonus Achieved:", averageAccuracy > 0.7);

    // Check if the average accuracy exceeds the threshold
    const bonusAchieved = averageAccuracy > 0.7;
    return bonusAchieved;
}


// ...existing code...

// First instruction page
const instructions1 = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: `
        <div class="instruction-section">
            <h1>Welcome to City Cabs!</h1>
            <p>As the dispatch coordinator, you are responsible for managing taxi routes across the city.</p>
            <p>At any given time, you will be offered two possible jobs, corresponding to two different pick-up and drop-off routes. Your task is to decide which job you want to accept.</p>
            <p>To help you get started, we'll go through a series of short practice trials.</p>
        </div>

        <div class="instruction-section">
            <h2>Press spacebar to continue </h2>
        </div>
    `,
    choices: [' '], // spacebar to continue
    on_load: function() {
        // Set initial city background to 'practice1.png'
        setCityBackground('practice1');
        grid.currentCity = 'practice1'; // Initialize the current city
    }  
};

const instructions2 = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {
        const fontSize = "20px"; // Define font size as a variable
        return `
            <div class="instruction-section" style="font-size: 20px;">
                <h1>Dispatch Instructions</h1>
                <p style="font-size: ${fontSize};">For each dispatch, you'll see two possible jobs marked in <span class="blue-text">blue</span> and <span class="green-text">green</span>. Each job has a passenger <img src="assets/people/blue_person.png" alt="Blue Passenger" width="23" height="23"> or <img src="assets/people/green_person.png" alt="Green Passenger" width="23" height="23"> at a pickup point, and a drop-off destination 🏠. The route of each job is marked with one of two letters:</p>
                <p style="font-size: ${fontSize};">- The letter <strong>F</strong> marks one job</p>
                <p style="font-size: ${fontSize};">- The letter <strong>J</strong> marks the other job</p>
                <p style="font-size: ${fontSize};">On each dispatch, these letters are randomly assigned to each job. To send out a taxi to one of these jobs, you need to press the corresponding key on your keyboard.</p>
                <p style="font-size: ${fontSize};">For any given choice, the lengths of the two possible jobs are the same, and you are paid the same wage by the company each day. However, some jobs are more costly than others, which you must pay yourself. This is because of tolls in the city...</p>
            </div>
            <div class="instruction-section" style="font-size: 20px;">
                <h1>Toll Intersections:</h1>
                <p style="font-size: ${fontSize};">Traffic in some parts of the city is busier than in other. This means that tolls apply at busy intersections. Visiting an intersection reveals whether or not you have to pay a toll there.</p>
                <p style="font-size: ${fontSize};">- <strong><span style="color: rgb(114, 114, 150);">Dark grey intersections</span></strong> have not been visited yet</p>
                <p style="font-size: ${fontSize};">- <strong><span style="color: #f87171;">Red intersections</span></strong> cost a $1 toll to pass through</p>
                <p style="font-size: ${fontSize};">- <strong><span style="color:rgb(194, 194, 229);">Light grey intersections</span></strong> are free with no tolls</p>
                <p style="font-size: ${fontSize};">Your goal is to complete all taxi jobs while minimizing total toll costs for your company.</p>
            </div>

            <div class="instruction-section" style="font-size: 20px;">
                <h2>Press spacebar to practise selecting a job.</h2>
            </div>
        `;
    },
    choices: [' '], // Spacebar to continue
};

const practice1SelectionTrial = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {
        

        // Determine the key assignment based on the trial index
        const keyAssignment = { blue: 'F', green: 'J' };
        const instruction = practice1TrialIndex === 0 
            ? `<h3>Please select the <span style="color: #5dadec; font-weight: bold;">BLUE</span> path by pressing the <span style="font-weight: bold;">${keyAssignment.blue}</span> key.</h3>`
            : `<h3>Please select the <span style="color:  #4ade80; font-weight: bold;">GREEN</span> path by pressing the <span style="font-weight: bold;">${keyAssignment.green}</span> key.</h3>`;
        
        // Store the assignment for this trial
        jsPsych.data.addProperties({
            blue_key: keyAssignment.blue,
            green_key: keyAssignment.green
        });
        const practice=true 
        
        return `
            <div class="instruction-section" style="text-align: center; margin-bottom: 20px; font-size: 18px; color: #3a3a3a;">
                <h3><strong>PRACTICE TRIAL:</strong><h3>
                ${instruction}
            </div>
            <div class="jobs-layout">
                <div class="current-job-section grid-fade-in">
                    <div class="current-job-container">
                        </div>
                        ${practice1Grid.createGridHTML(practice1TrialIndex, null, keyAssignment, true, true)}
                    </div>
                </div>
            </div>
        `;
        // return `
        //     <div class="instruction-section" style="text-align: center; margin-bottom: 20px; font-size: 18px; color: #3a3a3a;">
        //         <h3><strong>PRACTICE TRIAL:</strong><h3>
        //         ${instruction}
        //     </div>
        //     <div class="jobs-layout">
        //         <div class="current-job-section grid-fade-in">
        //             <div class="current-job-container">
        //                 </div>
        //                 ${practice1Grid.createGridHTML(practice1TrialIndex, null, keyAssignment, true, true)}
        //             </div>
        //         </div>
        //     </div>
        // `;
    },
    choices: function() {
        return practice1TrialIndex === 0 ? ['f'] : ['j'];
    },
    on_finish: function(data) {
        // Get the key assignment for this trial
        const keyAssignment = {
            blue: jsPsych.data.get().last(1).values()[0].blue_key,
            green: jsPsych.data.get().last(1).values()[0].green_key
        };
        
        // Determine the choice based on the key pressed
        let choice;
        if (data.response === 'f') {
            choice = keyAssignment.blue === 'F' ? 'blue' : 'green';
        } else if (data.response === 'j') {
            choice = keyAssignment.green === 'J' ? 'green' : 'blue';
        }
        
        // Record their choice
        data.choice = choice;
        data.practice_trial = true;
        
        // Check if they selected the correct path
        const correctChoice = (practice1TrialIndex === 0 && choice === 'blue') || (practice1TrialIndex === 1 && choice === 'green');
        data.correct = correctChoice;
        
        // Add swipe effect to visualize their selection
        const choiceElement = document.getElementById(`${choice}-choice`);
        const unchosenElement = document.getElementById(choice === 'blue' ? 'green-choice' : 'blue-choice');
        
        if (choiceElement && unchosenElement) {
            choiceElement.classList.add('choice-selected');
            unchosenElement.classList.add('choice-unselected');
        }
    
        // Replot the grid with only the chosen path
        const gridContainer = document.querySelector(".current-job-section");
        if (gridContainer) {
            gridContainer.innerHTML = `
                <div class="current-job-container">
                    <div class="cost-display-container">
                        <h2 class="cost-total">Total Tolls Paid:</h2>
                        <p id="total-cost" class="cost-total">-$0</p>
                        <p id="trial-cost" class="cost-trial hidden">-$0</p> 
                    </div>
                    ${practice1Grid.createGridHTML(practice1TrialIndex, choice, keyAssignment,true,true)}
                </div>
            `;
        }

        // Increment the practice trial index
        practice1TrialIndex++;
    }
};

// Practice animation trial
const practice1AnimationTrial = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {
        const lastTrialData = jsPsych.data.get().last(1).values()[0];
        const keyAssignment = { blue: lastTrialData.blue_key, green: lastTrialData.green_key };
        
        return `
            <div class="instruction-section" style="text-align: center; margin-bottom: 20px; font-size: 18px; color: #3a3a3a;">
                <h3><strong>PRACTICE TRIAL:</strong><h3>
                <h3>Watch the taxi follow the selected path.</h3>
            </div>
            <div class="jobs-layout">
                <div class="current-job-section">
                    <div class="current-job-container">
                        ${practice1Grid.createGridHTML(practice1TrialIndex - 1, lastTrialData.choice, keyAssignment,true,true)}
                    </div>
                </div>
            </div>
        `;
    },
    choices: "NO_KEYS",
    on_load: function() {
        const currentTrial = practice1Grid.getTrialInfo(practice1TrialIndex - 1);
        const lastTrialData = jsPsych.data.get().last(1).values()[0];

        if (!lastTrialData || !lastTrialData.choice) {
            console.error("No valid path choice found in practice trial.");
            return jsPsych.finishTrial();
        }

        const chosenPath = lastTrialData.choice === 'blue' ? currentTrial.path_A : currentTrial.path_B;
        const binaryCosts = practice1Grid.getBinaryCosts(`city_${currentTrial.city}_grid_${currentTrial.grid}`);

 
        practice1Grid.recordObservedCosts(chosenPath, binaryCosts);

        setTimeout(() => {
            animateAgent(chosenPath, binaryCosts, function() {
                jsPsych.finishTrial();
            });
        }, 100);
    }
};

const instructions3 = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {
        const n = grid.nTrials;
        const fontSize = "20px"; // Define font size as a variable
        return `
        <div class="instruction-section">
            <h1>Daily Shift:</h1>
            <p style="font-size: ${fontSize};">Each day, you will manage ${n} dispatches, meaning you have ${n} jobs to select.</p>
            <p style="font-size: ${fontSize};">All ${n} pairs of jobs will be presented on screen at once, side-by-side. Each dispatch takes place at a different time of the day and is marked with one of the following clock icons, displayed above the dispatch:</p>
            <p style="font-family: golemClocks; text-align: center; font-size: ${fontSize};">&#x00E6; &#x00DD; &#x0026; &#x263A;</p>
            <p style="font-size: ${fontSize};">You will move through these dispatches from left- to right-hand side of the screen. Your current dispatch is highlighted in <span style="color: #ece75d;">yellow</span>, while your past dispatches are <span style="color: rgb(138, 138, 184);">greyed out</span>.</p>
            <p style="font-size: ${fontSize};">You will first have 3 seconds to think about which job you would like to select. You can select your desired job once the dispatch grid turns yellow and the keys have been assigned to the paths - i.e. once 'F' or 'J' has been assigned to the green or blue job in your current dispatch.</p>
            <p style="font-size: ${fontSize};">You will have 7 seconds to select a job once the dispatch grid has turned yellow. If you fail to make a choice within this time limit, you will pay a fine of <span style="color: #f87171;">$10</span>.</p>
        </div>
        <div class="instruction-section">
            <h1>Toll Locations:</h1>
            <p style="font-size: ${fontSize};">The locations of tolls remain fixed throughout the day. Once you visit an intersection, you find out how busy it is, and hence whether or not you have to pay a toll whenever you reach that intersection again on the same day. This information is reflected in your upcoming dispatches, too.</p>
            <p style="font-size: ${fontSize};">This information may help you the rest of the day by allowing you to select jobs where you don’t have to pay any tolls.</p>
        </div>
        <div class="instruction-section">
            <h2 style="font-size: ${fontSize};">Press spacebar to practise your first full day.</h2>
        </div>
        `;
    },
    choices: [' '], // Spacebar to continue
};

const practiceFirstDayTrial = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {
        return `
            <div class="jobs-layout">
            <div class="upcoming-jobs-container grid ${practice2TrialIndex % practice2Grid.nTrials === 0 ? 'grid-fade-in' : ''}">
                ${practice2Grid.createAllJobsHTML(practice2TrialIndex, null, null, false, true)} 
            </div>
            </div>
        `;
    },
    choices: "NO_KEYS",
    trial_duration: 3000, // 
    on_finish: function() {
    }
};

const practice2PreSelectionTrial = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {
        // Reset total cost for practice if first practice trial
        if (practice2TrialIndex === 0) {
            totalCost = 0;
        }
        return `
            <div class="jobs-layout">
            <div class="upcoming-jobs-container grid">
                ${practice2Grid.createAllJobsHTML(practice2TrialIndex, null, null)} 
            </div>
            </div>
        `;
    },
    choices: "NO_KEYS",
    trial_duration: 3000, // Ends after 2 seconds
    on_finish: function() {
    }
};

const practice2SelectionTrial = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {
        // Randomly assign F and J to blue and green paths
        const keyAssignment = Math.random() < 0.5 ? 
            { blue: 'F', green: 'J' } : 
            { blue: 'J', green: 'F' };
        
        // Store the assignment for this trial
        jsPsych.data.addProperties({
            blue_key: keyAssignment.blue,
            green_key: keyAssignment.green
        });

        // Reset total cost for practice if first practice trial
        if (practice2TrialIndex === 0) {
            totalCost = 0;
        }
        const totalCostElement = document.getElementById("total-cost");
        if (totalCostElement) {
            totalCostElement.textContent = "$0";
        }
        
        return `
            <div class="jobs-layout">
                <div class="upcoming-jobs-container">
                    ${practice2Grid.createAllJobsHTML(practice2TrialIndex, null, keyAssignment)} 
                </div>
            </div>
        `;
    },
    choices: ['f', 'j'], 
    trial_duration: 7000, // Automatically ends after 5 seconds
    on_finish: function(data) {
        // Get the key assignment for this trial
        const keyAssignment = {
            blue: jsPsych.data.get().last(1).values()[0].blue_key,
            green: jsPsych.data.get().last(1).values()[0].green_key
        };
        
        // Determine the chosen path based on the key pressed
        let choice;
        if (data.response === 'f') {
            choice = keyAssignment.blue === 'F' ? 'blue' : 'green';
        } else if (data.response === 'j') {
            choice = keyAssignment.blue === 'J' ? 'blue' : 'green';
        } else {
            choice = 'nan'; // Log as 'nan' if no response is made
        }
        
        
        // Add "swipe" effect on selection
        const choiceElement = document.getElementById(`${choice}-choice`);
        const unchosenElement = document.getElementById(choice === 'blue' ? 'green-choice' : 'blue-choice');
        
        if (choiceElement && unchosenElement) {
            choiceElement.classList.add('choice-selected');
            unchosenElement.classList.add('choice-unselected');
        }
    
        // Replot the grid with only the chosen path
        const gridContainer = document.querySelector(".current-job-section");
        if (gridContainer) {
            gridContainer.innerHTML = practice2Grid.createGridHTML(practice2TrialIndex, choice, keyAssignment,true,true);
        }
        
        // // Store all the relevant data from the current trial
        const currentTrial = practice2Grid.getTrialInfo(practice2TrialIndex);
        data.choice = choice;
        data.trial = currentTrial.trial;
        data.city = currentTrial.city;
        data.grid_id = currentTrial.practice2Grid;
        data.path_chosen = choice;
        data.button_pressed = data.response;
        data.reaction_time_ms = data.rt;
        data.key_assignment = keyAssignment;
        data.path_A_expected_cost = currentTrial.path_A_expected_cost;
        data.path_B_expected_cost = currentTrial.path_B_expected_cost;
        data.path_A_actual_cost = currentTrial.path_A_actual_cost;
        data.path_B_actual_cost = currentTrial.path_B_actual_cost;
        data.path_A_future_overlap = currentTrial.path_A_future_overlap;
        data.path_B_future_overlap = currentTrial.path_B_future_overlap;
        data.abstract_sequence_A = JSON.stringify(currentTrial.abstract_sequence_A);
        data.abstract_sequence_B = JSON.stringify(currentTrial.abstract_sequence_B);
        data.dominant_axis_A = currentTrial.dominant_axis_A;
        data.dominant_axis_B = currentTrial.dominant_axis_B;
        data.better_path = currentTrial.better_path;
        const better_path_ID = currentTrial.better_path === 'a' ? 'blue' : currentTrial.better_path === 'b' ? 'green' : null;
        if (choice === better_path_ID) {
            data.chose_better_path = 1;
        } else {
            data.chose_better_path = 0;
        }

        // Include all columns from the current trial
        Object.keys(currentTrial).forEach(key => {
            data[key] = currentTrial[key];
        });

        // Include all trial info from the current trial
        Object.assign(data, currentTrial);
        
        // Add the trial data to jsPsych's data collection
        jsPsych.data.get().addToLast(data);

    }
};

const practice2AnimationTrial = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {
        const lastTrialData = jsPsych.data.get().last(1).values()[0];
        const keyAssignment = lastTrialData.key_assignment;
        
        return `
            <div class="jobs-layout">
                <div class="upcoming-jobs-container">
                    ${practice2Grid.createAllJobsHTML(practice2TrialIndex, lastTrialData.choice, keyAssignment)} 
                </div>
            </div>
        `;
    },
    choices: "NO_KEYS",
    on_load: function() {
        const currentTrial = practice2Grid.getTrialInfo(practice2TrialIndex);
        const lastTrialData = jsPsych.data.get().last(1).values()[0];

        // hacky
        currentTrialIndex = practice2TrialIndex;

        if (!lastTrialData || !lastTrialData.choice) {
            console.error("No valid path choice found. Restarting trial.");
            return jsPsych.finishTrial();
        }

        const chosenPath = lastTrialData.choice === 'blue' ? currentTrial.path_A : 
                   lastTrialData.choice === 'green' ? currentTrial.path_B : 
                   null;
        const binaryCosts = practice2Grid.getBinaryCosts(`city_${currentTrial.city}_grid_${currentTrial.grid}`);

        if (chosenPath !== null) {
            practice2Grid.recordObservedCosts(chosenPath, binaryCosts);
        }
        

        setTimeout(() => {
            animateAgent(chosenPath, binaryCosts, function() {
                jsPsych.finishTrial();
            });
        }, 100);
        
        // Increment the practice trial index
        practice2TrialIndex++;
    }
};

// show how the grid resets at the end of the day
const instructions4 = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: `
        <div class="instruction-section" style="z-index: 2001; position: relative;">
            <h1>New Day</h1>
            <p>At the start of a new day, the traffic in the city resets, meaning that the intersections where you do (or do not) have to pay a toll have reset.</p>
            <p>Watch the grid reset for the next day.</p>
            <p id="continue-text" style="display: none;">Press spacebar to continue dispatching.</p>
        </div>
        <div class="jobs-layout" style="z-index: 2001; position: relative;">
            <div id="grid-container" class="current-job-section"></div>
        </div>
    `,
    choices: "NO_KEYS", // Initially disable keypresses
    on_load: function() {
        // Show the current state of the grid for 1 second
        const gridContainer = document.getElementById('grid-container');
        gridContainer.innerHTML = practice2Grid.createGridHTML(practice2TrialIndex, 'none', null, false); // Render the current state of the grid without cost display, nor any current paths

        const blackCover = document.createElement('div');
        blackCover.style.position = 'fixed';
        blackCover.style.top = '0';
        blackCover.style.left = '0';
        blackCover.style.width = '100%';
        blackCover.style.height = '100%';
        blackCover.style.backgroundColor = 'black';
        blackCover.style.opacity = '0';
        blackCover.style.transition = 'opacity 1s ease-in-out';
        blackCover.style.zIndex = '1000'; // Ensure the black cover is behind the instructions and grid
        document.body.appendChild(blackCover);

        // Fade to full opacity
        setTimeout(() => {
            blackCover.style.opacity = '0.5';
        }, 10);

        // After 1s, set the new grid and fade back to transparency
        setTimeout(() => {
            gridContainer.innerHTML = practice2Grid.createBlankGridHTML(); // Render a blank grid
            gridContainer.style.opacity = "2";
            blackCover.style.opacity = '0'; // Fade back out
        }, 1000);

        // Remove the black cover after the transition is complete
        setTimeout(() => {
            document.body.removeChild(blackCover);

            // Show the continue text and enable spacebar
            document.getElementById("continue-text").style.display = "block";
            jsPsych.pluginAPI.getKeyboardResponse({
                callback_function: jsPsych.finishTrial, // Ends trial when spacebar is pressed
                valid_responses: [' '], // Spacebar
                rt_method: "performance",
                persist: false,
                allow_held_key: false
            });
        }, 2000);
    },
    on_finish: function() {
        practice2Grid.resetGrid(); // Reset the grid for the new set of trials
    }
};

// illustrate city change
const instructions5 = {
    type: jsPsychHtmlKeyboardResponse, 
    stimulus: function() {
        const n = grid.nGrids;
        return `
            <div class="instruction-section" style="z-index: 2000; position: relative;">
                <h1>New City</h1>
                <p>After ${n} days of working in one city, your taxi company starts operating in a new city.</p>
                <p>When you move cities, the background changes.</p>
            </div>
            <div class="instruction-section">
                <h2 id="continue-text" style="display: none;">Press spacebar to continue.</h2>
            </div>
        `;
    },
    choices: "NO_KEYS", // Initially disable keypresses
    on_load: function() {
        // Create a container for the animation
        let transitionContainer = document.createElement('div');
        transitionContainer.style.position = 'fixed';
        transitionContainer.style.top = '0';
        transitionContainer.style.left = '0';
        transitionContainer.style.width = '200%'; // Double width to fit both images
        transitionContainer.style.height = '100%';
        transitionContainer.style.zIndex = '1000';
        transitionContainer.style.display = 'flex';
        transitionContainer.style.transition = 'transform 1.5s ease-in-out';
        document.body.appendChild(transitionContainer);
        
        // Create old city element
        let oldCity = document.createElement('div');
        oldCity.style.width = '50%'; // Half of the container
        oldCity.style.height = '100%';
        oldCity.style.backgroundImage = `url('assets/cities/cropped/practice1.png')`;
        oldCity.style.backgroundSize = 'cover';
        oldCity.style.backgroundPosition = 'center';
        transitionContainer.appendChild(oldCity);
        
        // Create new city element
        let newCity = document.createElement('div');
        newCity.style.width = '50%'; // Half of the container
        newCity.style.height = '100%';
        newCity.style.backgroundImage = `url('assets/cities/cropped/practice2.png')`;
        newCity.style.backgroundSize = 'cover';
        newCity.style.backgroundPosition = 'center';
        transitionContainer.appendChild(newCity);
        
        // Force browser reflow before starting animation
        void transitionContainer.offsetWidth;
        
        // Start the slide animation
        transitionContainer.style.transform = 'translateX(-50%)';
        
        // After animation completes, set the new background and enable spacebar
        setTimeout(() => {
            setCityBackground('practice2');
            document.body.removeChild(transitionContainer);

            // Show the "Press spacebar to continue" text
            document.getElementById("continue-text").style.display = "block";

            // Enable spacebar input
            jsPsych.pluginAPI.getKeyboardResponse({
                callback_function: jsPsych.finishTrial, // Ends trial when spacebar is pressed
                valid_responses: [' '], // Spacebar
                rt_method: "performance",
                persist: false,
                allow_held_key: false
            });
        }, 1600);
    }
};
    


// illustrate column city
const instructions6 = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {
        const fontSize = "22px"; // Define font size as a variable
        document.body.style.zoom = "75%";
        return `
            <div class="instruction-section" style="margin: 10px;">
            <h1>How do you figure out where intersections with tolls (or no tolls) are?</h1>
            <p style="font-size: ${fontSize};">Each city has particular traffic properties, such that the busy streets tend to be related to one another in one of two ways.</p>
            </div>
            <div class="instruction-section" style="margin: 10px;">
            <h1>'Column cities'</h1>
            <p style="font-size: ${fontSize};">In column cities, traffic tends to run from north to south every day, meaning that tolls tend to be clustered in columns.</p>
            <p style="font-size: ${fontSize};">That is, a column may have a lot of tolls, or not many tolls.</p>
            <p style="font-size: ${fontSize};">The locations of these busy columns may change each day, but the city will always have this column-dependent feature.</p>
            </div>
            <div class="jobs-layout" >
            <div class="instruction-section" style="text-align: center; font-size: 20px; color: #3a3a3a; margin: 10px;">
            <h3><strong>Example day ${practice3TrialIndex + 1}</strong><h3>
            </div>
            <div id="grid-container" class="current-job-section"></div>
        `;
    },
    choices: [' '], // Wait for spacebar to continue
    on_load: function() {
        const gridContainer = document.getElementById('grid-container');
        const revealCosts = true; // Set to true to show costs
        console.log("Rendering grid for practice3TrialIndex:", practice3TrialIndex);
        gridContainer.innerHTML = practice3Grid.createBlankGridHTML(practice3TrialIndex, revealCosts); // Render a blank grid
        practice3TrialIndex++;
    },
};

// illustrate city change
const instructions7 = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: `
        <div class="instruction-section" style="z-index: 2000; position: relative;">
            <h1>How do you figure out where the intersections with tolls are?</h1>
            <p>Each city has particular traffic properties, such that the busy streets tend to be related to one another in one of two ways.</p>
    `,
    choices: "NO_KEYS", // No keypress required
    on_load: function() {
        // Create a container for the animation
        let transitionContainer = document.createElement('div');
        transitionContainer.style.position = 'fixed';
        transitionContainer.style.top = '0';
        transitionContainer.style.left = '0';
        transitionContainer.style.width = '200%'; // Double width to fit both images
        transitionContainer.style.height = '100%';
        transitionContainer.style.zIndex = '1000';
        transitionContainer.style.display = 'flex';
        transitionContainer.style.transition = 'transform 1.5s ease-in-out';
        document.body.appendChild(transitionContainer);

        // Create old city element
        let oldCity = document.createElement('div');
        oldCity.style.width = '50%'; // Half of the container
        oldCity.style.height = '100%';
        oldCity.style.backgroundImage = `url('assets/cities/cropped/practice2.png')`;
        oldCity.style.backgroundSize = 'cover';
        oldCity.style.backgroundPosition = 'center';
        transitionContainer.appendChild(oldCity);

        // Create new city element
        let newCity = document.createElement('div');
        newCity.style.width = '50%'; // Half of the container
        newCity.style.height = '100%';
        newCity.style.backgroundImage = `url('assets/cities/cropped/practice3.png')`;
        newCity.style.backgroundSize = 'cover';
        newCity.style.backgroundPosition = 'center';
        transitionContainer.appendChild(newCity);

        // Force browser reflow before starting animation
        void transitionContainer.offsetWidth;

        // Start the slide animation
        transitionContainer.style.transform = 'translateX(-50%)';
        
        // After animation completes, set the new background and finish the trial
        setTimeout(() => {
            setCityBackground('practice3');
            document.body.removeChild(transitionContainer);
            jsPsych.finishTrial(); // Automatically move to the next trial
        }, 1600);
    },
};

const instructions8 = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {
        return `
            <div class="instruction-section" style="z-index: 2000; position: relative;">
                <h1>How do you figure out where the intersections with tolls are?</h1>
                <p>Each city has particular traffic properties, such that the busy streets tend to be related to one another in one of two ways.</p>
            </div>
            <div class="instruction-section"> 
                <h1>'Row cities'</h1>
                <p>In row cities, the opposite is true: traffic tends to run from east to west every day, meaning that tolls tend to be clustered in rows.</p>
                <p>That is, a row may have a lot of tolls, or not many tolls.</p>
                <p>The locations of these busy rows may change each day, but the city will always have this row-dependent feature.</p>
            </div>
            <div class="jobs-layout">
                <div class="instruction-section" style="text-align: center;  font-size: 20px; color: #3a3a3a;">
                    <h3><strong>Example day ${practice4TrialIndex + 1}</strong><h3>
                </div>
                <div id="grid-container" class="current-job-section"></div>
            </div>
        `;
    },
    choices: [' '], // Wait for spacebar to continue
    on_load: function() {
        const gridContainer = document.getElementById('grid-container');
        const revealCosts = true; // Set to true to show costs
        gridContainer.innerHTML = practice4Grid.createBlankGridHTML(practice4TrialIndex, revealCosts); // Render a blank grid
        practice4TrialIndex++;
    },
};


const fullscreenTrial = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: `
        <div class="instruction-section">
            <h1>Enter Full Screen</h1>
            <p>The experiment will switch to full screen mode when you press the Enter key.</p>
            <p>Please also ensure the sound on your browser is turned on.</p>
            <p>This ensures the best experience during the experiment.</p>
        </div>
    `,
    choices: ["Enter"], // Listens for the Enter key
    on_finish: function() {
        document.documentElement.requestFullscreen(); // Forces full-screen mode
    }
};

// instructions review - i.e. ask participants if they want to review the instructions pages (instructions1-instructions8, without the practices)
const instructionsReview = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {
        document.body.style.zoom = "100%";
        return `
            <div class="instruction-section">
                <h1>Review Instructions</h1>
                <p>We will now ask you a few questions to check your understanding of the task. Before doing so, you have the opportunity to review the instructions.</p>
                <p>Would you like to see the instructions again?</p>
                <p>To review all the instructions from the beginning, press the backspace key.</p>
                <p>If you feel ready to continue, please press the spacebar.</p>
            </div>
        `;
    },
    choices: [' ', 'backspace'],
    on_finish: function(data) {
        if (data.response === 'backspace') {
            // Restart the experiment by reloading the page
            location.reload();
        }
    }
};

// Explanation of bonus
const instructions9 = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {
        const nDays = grid.nGrids; // Retrieve the number of days from grid.nGrids
        return `
            <div class="instruction-section">
                <h1>Bonus Payment</h1>
                <p>Remember: your aim is to minimise the cost paid each day by predicting which intersections will incur a toll, and hence by selecting jobs that you think will be least costly.</p>
                <p>At the end of the experiment, we will assess your performance by assessing how well you chose jobs that were the least costly on a randomly selected set of days and cities. This will determine how much bonus payment you receive.</p>
                <p>So, you should pay attention throughout the experiment - i.e. on every day, and in every city.</p>
                <p>Remember also: you will have 7 seconds to select a job once the current dispatch turns yellow, otherwise the trial will timeout and you will pay a fine of <span style="color: #f87171;">$10</span>. If you timeout too many times, the experiment will end and you will return to Prolific.</p>
                <p>When you are ready to begin the experiment, press the spacebar.</p>
            </div>
        `;
    },
    choices: [' '], // Spacebar to continue
};

// Create timeline
function createTimeline() {
    
    const timeline = [];

    // city assignments
    const numCities = data.env_costs.n_cities; // Assuming this is the number of cities
    createCityMapping(numCities);

    // Welcome message
    timeline.push(fullscreenTrial);
    timeline.push(instructions1);

    // // Practice selection
    timeline.push(instructions2);
    timeline.push(practice1SelectionTrial);
    timeline.push(practice1AnimationTrial);
    timeline.push(practice1SelectionTrial);
    timeline.push(practice1AnimationTrial);

    // Practice a full day
    timeline.push(instructions3);
    timeline.push(practiceFirstDayTrial);
    for (let i = 0; i < grid.nTrials; i++) {
        timeline.push(practice2PreSelectionTrial);
        timeline.push(practice2SelectionTrial);
        timeline.push(practice2AnimationTrial);
    }
    timeline.push(practiceGridFeedback);

    // Animation to show grid resetting, and then another day
    timeline.push(instructions4);
    timeline.push(practiceFirstDayTrial);
    for (let i = 0; i < grid.nTrials; i++) {
        timeline.push(practice2PreSelectionTrial);
        timeline.push(practice2SelectionTrial);
        timeline.push(practice2AnimationTrial);
    }
    timeline.push(practiceGridFeedback);

    // New city animation
    timeline.push(instructions5);

    for (let i = 1; i <= grid.nGrids; i++) {
        timeline.push(instructions6);
    }
    timeline.push(instructions7);
    for (let i = 1; i <= grid.nGrids; i++) {
        timeline.push(instructions8);
    }

    // Add the option to review the instructions
    timeline.push(instructionsReview);
    
    // Understanding checks
    const quizTrials = createQuizTrials(jsPsych);
    timeline.push(...quizTrials);

    // bonus message
    timeline.push(instructions9)

    // Add the first grid message
    timeline.push(firstGridMessage);

    // Loop through all trials and add them to the timeline
    for (let i = 0; i < grid.trialInfo.length; i++) {
        if (i % grid.nTrials === 0) {
             if (i !== 0) {
                 timeline.push(newDayMessage);
                // add new grid message if the city changes, i.e. if i is a multiple of nTrials*nGrids
                if (i % (grid.nTrials * grid.nGrids) === 0) {
                    timeline.push(timeoutCheck);
                    timeline.push(newCityMessage);
                }
            }
            timeline.push(firstDayTrial)
        }
        timeline.push(pathPreSelectionTrial);
        timeline.push(pathSelectionTrial);
        timeline.push(pathAnimationTrial);
    }
    // Add the end and bonus message
    timeline.push(end);
    timeline.push(bonus);
    return timeline;
}

function downloadTrialData() {
    // Get all path selection trial data
    const pathData = jsPsych.data.get().filter({trial_type: 'html-keyboard-response'}).values();
    
    // Format the data for CSV
    const trialData = pathData.map(trial => {
        // Only include trials where a choice was made
        if (trial.choice) {
            if (trial.choice) {
                return trial; // Return the entire trial object
            }
        }
        return null;
    }).filter(item => item !== null);

    // Convert the data to CSV format
    const csvHeaders = "trial,city,grid_id,path_chosen,button_pressed,reaction_time_ms,context,grid,cityID,path_A_expected_cost,path_B_expected_cost,path_A_actual_cost,path_B_actual_cost,path_A_future_overlap,path_B_future_overlap,abstract_sequence_A,abstract_sequence_B,dominant_axis_A,dominant_axis_B,better_path,chose_better_path\n";
    const csvRows = trialData.map(trial => 
        `${trial.trial},${trial.city},${trial.grid_id},${trial.path_chosen},${trial.button_pressed},${trial.reaction_time_ms},${trial.context},${trial.grid},${trial.cityID},${trial.path_A_expected_cost},${trial.path_B_expected_cost},${trial.path_A_actual_cost},${trial.path_B_actual_cost},${trial.path_A_future_overlap},${trial.path_B_future_overlap},"${trial.abstract_sequence_A}","${trial.abstract_sequence_B}",${trial.dominant_axis_A}",${trial.dominant_axis_B}",${trial.better_path},${trial.chose_better_path}`
    ).join("\n");
    const csvContent = csvHeaders + csvRows;

    // Create a Blob with the CSV data
    const dataBlob = new Blob([csvContent], {type: 'text/csv'});
    
    // Create a download link and trigger it
    const url = URL.createObjectURL(dataBlob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'path_selection_data.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// Start experiment when the page loads
function initializeExperiment() {


    // Run the instruction timeline first
    const timeline = createTimeline();
    jsPsych.run(timeline);

}