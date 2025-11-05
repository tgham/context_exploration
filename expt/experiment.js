// Initialize jsPsych
const jsPsych = initJsPsych({
    on_finish: function() {
        // Show a waiting message
        document.body.innerHTML = `
            <div class="instruction-section">
                <h2>Please wait while you are redirected to Prolific.</h2>
                <h2 style="font-weight: bold; ">DO NOT CLOSE YOUR BROWSER until you have returned to Prolific. This may take a few seconds...</h2>
            </div>
        `;

        // Get participant data and send it
        // var ppt_data = jsPsych.data.get().json();
        // console.log('experiment complete');

        // Define the variables you want to keep
        // var keep_vars = [
        //     "pid", "trial", "city", "path_chosen", "button_pressed", "reaction_time_ms","html-keyboard-response","choice",
        //     "context", "grid", "path_A_expected_cost", "path_B_expected_cost",
        //     "path_A_actual_cost", "path_B_actual_cost", "path_A", "path_B",
        //     "path_A_future_overlap", "path_B_future_overlap",
        //     "path_A_future_row_overlap", "path_B_future_row_overlap",
        //     "path_A_future_col_overlap", "path_B_future_col_overlap",
        //     "path_A_future_row_and_col_overlap", "path_B_future_row_and_col_overlap",
        //     "path_A_future_rel_overlap", "path_B_future_rel_overlap",
        //     "path_A_future_irrel_overlap", "path_B_future_irrel_overlap",
        //     "abstract_sequence_A", "abstract_sequence_B",
        //     "dominant_axis_A", "dominant_axis_B",
        //     "better_path", "chose_better_path", "bonusAchieved", "expt_info_filename"
        // ];

        // Filter the jsPsych data down to just those variables
        // var ppt_data = jsPsych.data.get().filterColumns(keep_vars).json();
        // console.log('experiment complete (filtered dataset)');
        // var ppt_data = jsPsych.data.get()
        //     .filter({ trial_type: 'html-keyboard-response' }) // only keep choice trials
        //     .json();
        // console.log('experiment complete (filtered dataset)');

        var ppt_data = JSON.stringify(
            jsPsych.data.get()
            .filter({ trial_type: 'html-keyboard-response' })
            .values()
            .map(({ stimulus, ...rest }) => rest)
        );
        console.log('experiment complete (filtered dataset)');
        
        
        send_complete(subject_id, ppt_data)
            .then(() => {
                console.log('Data successfully sent to completion endpoint');
                if (bonusAchieved) {
                    window.location.replace("https://app.prolific.com/submissions/complete?cc=C19WDNCC");
                } else {
                    window.location.replace("https://app.prolific.com/submissions/complete?cc=C1HB0QAK");
                }
            })
            .catch(error => {
                console.error('Failed to send completion data:', error);
                // Still redirect after a delay in case of error
                document.body.innerHTML += `
                    <div class="instruction-section">
                        <h2>Note: There might have been an issue saving your data, but you will still be redirected shortly.</h2>
                    </div>
                `;
                setTimeout(() => {
                    if (bonusAchieved) {
                        window.location.replace("https://app.prolific.com/submissions/complete?cc=C19WDNCC");
                    } else {
                        window.location.replace("https://app.prolific.com/submissions/complete?cc=C1HB0QAK");
                    }
                }, 5000);
            });
    }
});

import { createQuizTrials } from './test.js';

// decide whether we're doing this properly or not...
let test = false;
let subject_id = null;
let sequence = null;
let data = null;
let grid = null;

// just test with this...
if (test) {

    // var subject_id = 1
    jsPsych.data.addProperties({
        subject_id: subject_id,
    });
    var ppt_data = jsPsych.data.get().json();
    send_incomplete(subject_id, ppt_data);
    console.log('debugging with subject_id 1');
    fetch('assets/trial_sequences/expt_2/expt_info/expt_2_info_1.json')
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
            window.location.replace("error.html");
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
        window.location.replace("error.html");
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

// Global object to store preloaded images
const preloadedImages = {};

// Function to preload images
function preloadImages(imagePaths) {
    return Promise.all(
        imagePaths.map(path => {
            return new Promise((resolve, reject) => {
                const img = new Image();
                img.src = path;

                // Set attributes for the image
                if (path.includes('blue_person.png')) {
                    img.alt = "Blue Start";
                    img.width = 25;
                    img.height = 25;
                } else if (path.includes('green_person.png')) {
                    img.alt = "Green Start";
                    img.width = 25;
                    img.height = 25;
                }

                img.onload = () => {
                    preloadedImages[path] = img; // Store the loaded image
                    resolve();
                };
                img.onerror = reject;
            });
        })
    ).then(() => {
        console.log('All images preloaded successfully.');
    }).catch(error => {
        console.error('Error preloading images:', error);
        throw error;
    });
}

// Preload the images at the start
preloadImages([
    'assets/people/blue_person.png',
    'assets/people/green_person.png'
]).then(() => {
    console.log('Avatars are ready for use.');
});

// Function to calculate and apply the scaling factor
let zoomFactor;
function applyScreenScaling() {
    
    // Define the reference dimensions (MacBook Pro 16" M2)
    const baseWidth = 3456 / 2;
    const baseHeight = 2234 / 2;

    const screenWidth = window.screen.width;
    const screenHeight = window.screen.height;

    const widthRatio = screenWidth / baseWidth;
    const heightRatio = screenHeight / baseHeight;

    // Start with a base zoom factor
    const baseZoomFactor = 0.85;

    // Adjust the zoom factor based on the screen proportions
    zoomFactor = baseZoomFactor * Math.min(widthRatio, heightRatio);

    document.body.style.zoom = zoomFactor;
    
    console.log(`User screen: ${screenWidth}x${screenHeight}`);
    console.log(`Reference screen: ${baseWidth}x${baseHeight}`);
    console.log(`Base zoom factor: ${baseZoomFactor}`);
    console.log(`Applied scaling factor: ${zoomFactor.toFixed(3)}`);
}



// consent etc
const informedConsentForm = `
    <div style="max-width: 800px; margin: auto; padding: 20px; font-family: Arial, sans-serif; line-height: 1.6; text-align: left;">
        <h2 style="text-align: center; color:rgb(255, 255, 255);">Informed Consent Form for Online Experiments on Cognitive Control</h2>
        <p>This is a psychology experiment being conducted by Dr. Peter Dayan, director of the Max Planck Institute for Biological Cybernetics, and the members of his lab. In order to consent to participate, you MUST meet the following criteria:</p>
        <ul style="margin: 20px 0; padding-left: 20px;">
            <li>18 years of age or older.</li>
            <li>Fluent speaker of English.</li>
            <li>Have not previously participated in this experiment.</li>
        </ul>
        <p>This study is designed to look at how people learn to make decisions to accomplish their goals. In this task, you will be asked to make choices, play games, and answer questions related to those games. The study will take about 40 minutes and will pay £6 plus a performance-dependent bonus of £2. The performance bonus is explained in more detail in the instructions that follow.</p>
        <p>Your participation in this research is voluntary. You may refrain from answering any questions that make you uncomfortable and may withdraw your participation at any time without penalty by exiting this task and alerting the experimenter. You may choose not to complete certain parts of the task or answer certain questions. You may contact us at the address provided below if you have additional questions or concerns.</p>
        <p>Other than monetary compensation, participating in this study will provide no direct benefits to you. But we hope that this research will benefit society at large by contributing towards establishing a scientific foundation for improving people’s learning and cognitive control abilities.</p>
        <p>Your online username may be connected to your individual responses, but we will not be asking for any additional personally identifying information, and we will handle responses as confidentially as possible. We cannot however guarantee the confidentiality of information transmitted over the Internet. We will be keeping de-identified data collected as part of this experiment indefinitely. Data used in scientific publications will remain completely anonymous.</p>
        <p>If you have any questions about the study, feel free to contact our lab. Dr. Dayan and his lab members can be reached at <a href="mailto:kyblab.tuebingen@gmail.com">kyblab.tuebingen@gmail.com</a>.</p>
        <p>By selecting the “consent” option below, I acknowledge that I am 18 or older, that I am a fluent speaker of English, that I have read this consent form, and that I agree to take part in the research.</p>
        <div style="text-align: center; margin-top: 20px;">
            <button id="consent-given" style="background-color: #2C3E50; color: white; padding: 10px 20px; border: none; cursor: pointer; margin-right: 10px;">I consent to participate</button>
            <button id="consent-rescinded" style="background-color: #E74C3C; color: white; padding: 10px 20px; border: none; cursor: pointer;">I do not consent to participate</button>
        </div>
    </div>
`;

const dataProtectionForm = `
<div style="max-width: 800px; margin: auto; padding: 20px; font-family: Arial, sans-serif; line-height: 1.6; text-align: left;">
    <h2 style="text-align: center; color:rgb(255, 255, 255);">Data Protection Form for Online Experiments on Cognitive Control</h2>
    <p>Please click <a href="data-protection-form.pdf" target="_blank" style="color: #2980B9;">here</a> to view our Data Protection Information Sheet. It is yours to keep.</p>
    <p>I have received and taken note of the Data Protection Information Sheet for this study. In doing so, I had sufficient time and opportunity to ask questions about data protection and reconsider my participation in the study. I am aware that:</p>
    <ul style="margin: 20px 0; padding-left: 20px;">
        <li>The processing and use of the collected data occurs in a pseudoanonymised form within the scope of the legally prescribed provisions. As a general rule, the storage occurs in the form of answered questionnaires, as well as electronic data, for a duration of 10 years or longer, if this is required by the purpose of the study.</li>
        <li>By providing further personal data in pseudoanonymised form, collected personal data may be used for the preparation of anonymised scientific research work and may also be published and used in an anonymised form in medical journals and scientific publications, so that a direct assignment to my person cannot be established.</li>
        <li>The information obtained during the course of this study may also be sent in an anonymised form to cooperation partners within the scope of the European General Data Protection Regulation for scientific purposes and to cooperation partners outside of the European Union, i.e. to countries with a lower data protection level (this also applies to the USA).</li>
        <li>The data collected within the scope of the study can also be used and processed in the future inside of the Max Planck Institute.</li>
    </ul>
    <p>I was informed about my rights. In particular, that at any time:</p>
    <ul style="margin: 20px 0; padding-left: 20px;">
        <li>I can withdraw this declaration of consent.</li>
        <li>I can request information about my stored data and request the correction or blocking of data.</li>
        <li>If I choose to stop participating in this study, I can request that any of my personal data associated with this study are immediately deleted or anonymised.</li>
        <li>I can request that my personal data are handed out to me or to third parties (if technically feasible).</li>
    </ul>
    <p>I hereby declare that:</p>
    <ul style="margin: 20px 0; padding-left: 20px;">
        <li>I have been adequately informed about the collection and processing of my personal data and rights.</li>
        <li>I consent to the collection and processing of personal data within the scope of the study and its pseudoanonymised disclosure, so that only persons conducting the study can establish a link between the data and my person.</li>
    </ul>
    <p>Your online username may be connected to your individual responses, but we will not be asking for any additional personally identifying information, and we will handle responses as confidentially as possible.</p>
    <p>We cannot however guarantee the confidentiality of information transmitted over the internet. We will keep de-identified data collected as part of this experiment indefinitely. Data used in scientific publications will remain completely anonymous.</p>
    <div class="consent-checkboxes" style="margin-top: 20px;">
        <input type="checkbox" id="checkbox-y" style="margin-right: 10px;">
        <label for="checkbox-y" style="font-weight: normal;">I agree to participate in this experiment.</label>
    </div>
    <div class="consent-checkboxes" style="margin-top: 10px;">
        <input type="checkbox" id="checkbox-z" style="margin-right: 10px;">
        <label for="checkbox-z" style="font-weight: normal;">I consent to the use of my data as described in the Data Protection Information Sheet and confirm having received a copy of the Data Protection Sheet.</label>
    </div>
    <div class="consent-checkboxes" style="margin-top: 10px;">
        <input type="checkbox" id="checkbox-x" style="margin-right: 10px;">
        <label for="checkbox-x" style="font-weight: normal;">I consent to data transfer from the MPI for Biological Cybernetics encrypted database to the project-related collaborators: inside of the Max Planck Society and affiliated research institutes, or at partnering institutions like the University of Tuebingen and New York University.</label>
    </div>
    <div style="text-align: center; margin-top: 20px;">
        <button id="submit-button" class="button small-button disabled" style="background-color: #2C3E50; color: white; padding: 10px 20px; border: none; cursor: pointer;" disabled>Submit</button>
        <button id="data-consent-rescinded" style="background-color: #E74C3C; color: white; padding: 10px 20px; border: none; cursor: pointer; margin-left: 10px;">I do not consent to all of the above</button>
    </div>
</div>
`;

const informedConsentTrial = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: informedConsentForm,
    response_ends_trial: false,
    on_load: function() {
        document.body.style.overflowY = "auto";
        document.documentElement.style.overflowY = "auto";
        document.getElementById('consent-given').addEventListener('click', function() {
            jsPsych.finishTrial();
        });
        document.getElementById('consent-rescinded').addEventListener('click', function() {
            alert('You chose not to consent to participate. Please return your submission on Prolific.');
        });
    }
};

const dataProtectionTrial = {
    type: jsPsychHtmlKeyboardResponse, // This is correct for both v6 and v7
    stimulus: dataProtectionForm,
    response_ends_trial: false, // Prevents trial from ending with a key press
    on_load: () => {
        const submitButton = document.getElementById('submit-button');
        const checkboxes = Array.from(document.querySelectorAll('.consent-checkboxes input'));

        checkboxes.forEach(checkbox => {
            checkbox.addEventListener('change', () => {
                const allChecked = checkboxes.every(cb => cb.checked);
                submitButton.disabled = !allChecked;
                submitButton.classList.toggle('disabled', !allChecked);
            });
        });

        submitButton.addEventListener('click', () => {
            if (!submitButton.disabled) {
                jsPsych.finishTrial(); // Ends the trial and proceeds
            }
        });

        document.getElementById('data-consent-rescinded').addEventListener('click', () => {
            alert('You chose not to consent to participate. Please return your submission on Prolific.');
        });
    }
};


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
                <div class="grid-container" style="grid-template-columns: repeat(${gridSize}, 40px);">
        `;

        // preload avatars
        // const bluePerson = preloadedImages['assets/people/blue_person.png'];
        // const greenPerson = preloadedImages['assets/people/green_person.png'];
        const bluePerson = '✋';
        const greenPerson = '✋';

    
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
                        pathClass = 'half-half-blue-green';

                        // show both keys in each of their colours
                        const colorOfP = getColorForKey(keyAssignment, 'P') || 'blue';
                        const colorOfQ = getColorForKey(keyAssignment, 'Q') || (colorOfP === 'blue' ? 'green' : 'blue');
                        content =    `<span class="${colorOfP === 'blue' ? 'blue-text' : 'green-text'}"  style="font-size: 1.5rem;"
                                    >P</span>` +
                                     `<span class="${colorOfQ === 'blue' ? 'blue-text' : 'green-text'}"  style="font-size: 1.5rem;"
                                     >Q</span>`;                        

                    } else if (isPathA) {
                        pathClass = 'blue-path';
                        content = `<span class="blue-text">${keyAssignment.blue}</span>`;
                    } else if (isPathB) {
                        pathClass = 'green-path';
                        content = `<span class="green-text">${keyAssignment.green}</span>`;
                    }
                } else {
                    // Fall back to stars if no key assignment provided
                    if (isOverlap) {
                        // const randomChoice = Math.random() < 0.5;
                        // pathClass = randomChoice ? 'blue-path' : 'green-path';
                        // content = randomChoice ? '<span class="green-text" style="font-size: 2rem;">+</span>' : '<span class="blue-text" style="font-size: 2rem;">+</span>';
                        content = '<span class="plus-split blue-green" style="font-size: 2rem;">+</span>';
                    } else if (isPathA || isPathB) {
                        // content = '+';
                        content = '<span class="' + (isPathA ? 'blue-text' : 'green-text') + '" style="font-size: 2rem;">+</span>';
                    }
                }
    
                if (isStartA) {
                    gridHTML += `<div class="grid-cell start blue-path ${observedClass}" id="${cellId}">`;
                    gridHTML += bluePerson; 
                    gridHTML += `</div>`;
                } else if (isStartB) {
                    gridHTML += `<div class="grid-cell start green-path ${observedClass}" id="${cellId}">`;
                    gridHTML += greenPerson; 
                    gridHTML += `</div>`;
                } else if (isGoalA) {
                    gridHTML += `<div class="grid-cell goal blue-path ${observedClass}" id="${cellId}">🏠</div>`;
                } else if (isGoalB) {
                    gridHTML += `<div class="grid-cell goal green-path ${observedClass}" id="${cellId}">🏠</div>`;
                } else if (isPathA || isPathB || isOverlap) {
                    gridHTML += `<div class="grid-cell ${observedClass} ${pathClass}" id="${cellId}" style="font-size: 2rem;-webkit-text-stroke: 0.5px black; text-shadow: 1px 1px 1px black;">${content}</div>`;
                } else {
                    gridHTML += `<div class="grid-cell ${observedClass} ${pathClass}" id="${cellId}"></div>`;
                }
            }
        }   
        gridHTML += `</div></div>`;
    
        return gridHTML;
    };

    // method for plotting a grid, either blank or with all costs revealed
    createBlankGridHTML(trialIndex = null, revealCosts = false, feedback=false, revealedCosts = 'all') {

        // let gridHTML = `
        //     <div class="current-job-container">
        // `;
        let gridHTML = `
            <div class="upcoming-jobs-actual-container">
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
            <div class="upcoming-job">  
                <div class="upcoming-grid" style="grid-template-columns: repeat(${this.gridSize}, 40px);">
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
                    if (revealedCosts === 'all') {
                        const cost = binaryCosts ? binaryCosts[row][col] : 0; // Safely access binaryCosts
                        const costClass = cost === -1 ? 'observed-cost' : 'observed-no-cost';
                        gridHTML += `<div class="grid-cell ${costClass}" id="${cellId}"></div>`;
                    } else if (revealedCosts === 'observed') {
                        const cost = this.observedCosts[`${row}-${col}`];
                        const costClass = cost !== undefined ? 
                            (cost === -1 ? 'observed-cost' : 'observed-no-cost') : '';
                        gridHTML += `<div class="grid-cell ${costClass}" id="${cellId}"></div>`;
                    }
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

        // preload avatars
        const bluePerson = preloadedImages['assets/people/blue_person.png'];
        const greenPerson = preloadedImages['assets/people/green_person.png'];

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
                        content = randomChoice ? '<span class="green-text" style="font-size: 2rem;">+</span>' : '<span class="blue-text" style="font-size: 2rem;">+</span>';
                    } else if (isPathA) {
                        pathClass = 'blue-path';
                        content = '+';
                    } else if (isPathB) {
                        pathClass = 'green-path';
                        content = '+';
                    }

                    if (isStartA) {
                        upcomingHTML += `<div class="upcoming-cell ${observedClass} blue-path" data-row="${row}" data-col="${col}">`;
                        upcomingHTML += bluePerson.outerHTML; // Use the preloaded blue person image
                        upcomingHTML += `</div>`;
                    } else if (isStartB) {
                        upcomingHTML += `<div class="upcoming-cell ${observedClass} green-path" data-row="${row}" data-col="${col}">`;
                        upcomingHTML += greenPerson.outerHTML; // Use the preloaded green person image
                        upcomingHTML += `</div>`;
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
    createAllJobsHTML(currentTrialIndex, selectedPath=null, keyAssignment=null, feedback=false, firstDay=false, showPink=true, restrictPink=null) {
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
        
        // Collect all upcoming paths for the current trial
        const upcomingPaths = new Set();
        if (!feedback && showPink) {
            let pinkEndIndex = restrictPink === null ? currentGridEndIndex : currentGridStartIndex + restrictPink;
            for (let i = currentTrialIndex + 1; i <= pinkEndIndex; i++) {
                const upcomingTrial = this.getTrialInfo(i);
                upcomingTrial.path_A.forEach(coord => upcomingPaths.add(`${coord[0]}-${coord[1]}`));
                upcomingTrial.path_B.forEach(coord => upcomingPaths.add(`${coord[0]}-${coord[1]}`));
            }
        }

        for (let i = 0; i < totalTrialsInGrid; i++) {
            const previewIndex = currentGridStartIndex + i;
            const trial = this.getTrialInfo(previewIndex);
            const clockCharacter = clockCharacters[i]; // Cycle through the characters based on the trial index
            
            if (!firstDay) {
                if (!feedback) {
                    // if (previewIndex < currentTrialIndex || (restrictPink !== null && i !== restrictPink && i !== 0)) {
                    if (previewIndex < currentTrialIndex || (restrictPink !== null && i > restrictPink)) {
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
                                <div class="clock-container" style="font-size: 50px; text-align: center; margin-bottom: 10px; background-color: #ece75d;">
                                    ${clockCharacter}
                                </div>
                                <div class="upcoming-grid" style="grid-template-columns: repeat(${this.gridSize}, 30px); grid-auto-rows: 30px;">
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

            // preload avatars
            // const bluePerson = preloadedImages['assets/people/blue_person.png'];
            // const greenPerson = preloadedImages['assets/people/green_person.png'];
            const bluePerson = '✋';
            const greenPerson = '✋';


            for (let row = 0; row < this.gridSize; row++) {
                for (let col = 0; col < this.gridSize; col++) {
                    const cellId = `cell-${row}-${col}-trial-${trial.trial}`;
                    let isStartA, isStartB, isGoalA, isGoalB, isPathA, isPathB;

                    // Check if this cell is in upcoming paths (only for current trial)
                    const isUpcomingPath = previewIndex === currentTrialIndex && upcomingPaths.has(`${row}-${col}`);

                    if (previewIndex > currentTrialIndex) {
                        isStartA = row === trial.start_A[0] && col === trial.start_A[1];
                        isStartB = row === trial.start_B[0] && col === trial.start_B[1];
                        isGoalA = row === trial.goal_A[0] && col === trial.goal_A[1];
                        isGoalB = row === trial.goal_B[0] && col === trial.goal_B[1];
                        isPathA = trial.path_A.some(coord => coord[0] === row && coord[1] === col);
                        isPathB = trial.path_B.some(coord => coord[0] === row && coord[1] === col);
                    } else if (previewIndex === currentTrialIndex) {
                        isStartA = selectedPath !== 'green' && selectedPath !== 'none' && row === trial.start_A[0] && col === trial.start_A[1];
                        isStartB = selectedPath !== 'blue' && selectedPath !== 'none' && row === trial.start_B[0] && col === trial.start_B[1];
                        isGoalA = selectedPath !== 'green' && selectedPath !== 'none' && row === trial.goal_A[0] && col === trial.goal_A[1];
                        isGoalB = selectedPath !== 'blue' && selectedPath !== 'none' && row === trial.goal_B[0] && col === trial.goal_B[1];
                        isPathA = selectedPath !== 'green' && selectedPath !== 'none' && trial.path_A.some(coord => coord[0] === row && coord[1] === col);
                        isPathB = selectedPath !== 'blue' && selectedPath !== 'none' && trial.path_B.some(coord => coord[0] === row && coord[1] === col);
                    } else if (previewIndex < currentTrialIndex) {
                            isStartA = false;
                            isStartB = false;
                            isGoalA = false;
                            isGoalB = false;
                            isPathA = false;
                            isPathB = false;
                    }
                    const observedCost = this[`observedCosts${i}`][`${row}-${col}`];
                    const observedClass = observedCost !== undefined ?
                        (observedCost === -1 ? 'observed-cost' : 'observed-no-cost') : '';
                        
                    // Handle overlapping paths
                    const isOverlap = isPathA && isPathB;
                    let pathClass = '';
                    let content = '';

                    // Determine border class based on path combinations (for current trial only)
                    // if (previewIndex === currentTrialIndex && !isStartA && !isStartB && !isGoalA && !isGoalB) {
                    if (previewIndex === currentTrialIndex) {
                        // Determine which paths this cell is on
                        const onBlue = isPathA;
                        const onGreen = isPathB;
                        const onUpcoming = isUpcomingPath;
                        
                        // Generate random order for borders when there are multiple paths
                        if (onBlue && onGreen && onUpcoming) {
                            // All three paths - randomize order
                            // const orders = [
                            //     // 'blue-green-magenta',
                            //     // 'blue-magenta-green',
                            //     // 'green-blue-magenta',
                            //     // 'green-magenta-blue',
                            //     'magenta-blue-green',
                            //     'magenta-green-blue'
                            // ];
                            // pathClass = orders[Math.floor(Math.random() * orders.length)];
                            if (i % 2 === 0) {
                                pathClass = 'magenta-blue-green'
                            } else {
                                pathClass = 'magenta-green-blue'
                            }
                        } else if (onBlue && onGreen) {
                            // Blue and green only
                            if (i % 2 === 0) {
                                pathClass = 'half-half-blue-green';
                            } else {
                                pathClass = 'half-half-green-blue';
                            }
                        } else if (onBlue && onUpcoming) {
                            
                            // Blue and magenta, with corner ordering depending on trial index
                            // if (i % 2 === 0) {
                            //     pathClass = 'half-half-blue-magenta';
                            // } else {
                            //     pathClass = 'half-half-magenta-blue';
                            // }

                            // or, just keep magenta in bottom corner
                            pathClass = 'half-half-blue-magenta';
                        } else if (onGreen && onUpcoming) {
                            
                            // Green and magenta, with corner ordering depending on trial index
                            // if (i % 2 === 0) {
                            //     pathClass = 'half-half-green-magenta';
                            // } else {
                            //     pathClass = 'half-half-magenta-green';
                            // }

                            // or, just keep magenta in bottom corner
                            pathClass = 'half-half-green-magenta';
                        } else if (onBlue) {
                            pathClass = 'blue-path';
                        } else if (onGreen) {
                            pathClass = 'green-path';
                        } else if (onUpcoming) {
                            pathClass = 'magenta-path';
                        }
                    }

                    // Determine content (key assignments or default symbols)
                    if (previewIndex === currentTrialIndex) {
                        if (keyAssignment) {
                            
                            // With key assignment: 
                            if (isOverlap) {
                                
                                // show both keys in each of their colours
                                const colorOfP = getColorForKey(keyAssignment, 'P') || 'blue';
                                const colorOfQ = getColorForKey(keyAssignment, 'Q') || (colorOfP === 'blue' ? 'green' : 'blue');
                            
                                // can also change font size of 'PQ' if needed - style="font-size: 1.2rem;"
                                content =    `<span class="${colorOfP === 'blue' ? 'blue-text' : 'green-text'}"  style="font-size: 1.2rem;"
                                            >P</span>` +
                                             `<span class="${colorOfQ === 'blue' ? 'blue-text' : 'green-text'}"  style="font-size: 1.2rem;"
                                             >Q</span>`;

                                // or, show single '+' symbol without P/Q labels
                                // if (i % 2 === 0) {
                                //     // was: two pluses
                                //     // content = '<span class="blue-text" style="font-size: 2rem;">+</span><span class="green-text" style="font-size: 2rem;">+</span>';
                                //     content = '<span class="plus-split blue-green" style="font-size: 2rem;">+</span>';
                                // } else {
                                //     // content = '<span class="green-text" style="font-size: 2rem;">+</span><span class="blue-text" style="font-size: 2rem;">+</span>';
                                //     content = '<span class="plus-split green-blue" style="font-size: 2rem;">+</span>';
                                // }
                                
                            } else if (isPathA && isUpcomingPath) {
                                // Blue path that's also on upcoming path
                                content = `<span class="blue-text">${keyAssignment.blue}</span>`;
                            } else if (isPathB && isUpcomingPath) {
                                // Green path that's also on upcoming path
                                content = `<span class="green-text">${keyAssignment.green}</span>`;
                            } else if (isPathA) {
                                content = keyAssignment.blue;
                            } else if (isPathB) {
                                content = keyAssignment.green;
                            }
                        } else {
                            // Without key assignment: show default symbols
                            if (isOverlap) {
                                if (i % 2 === 0) {
                                    // was: two pluses
                                    // content = '<span class="blue-text" style="font-size: 2rem;">+</span><span class="green-text" style="font-size: 2rem;">+</span>';
                                    content = '<span class="plus-split blue-green" style="font-size: 2rem;">+</span>';
                                } else {
                                    // content = '<span class="green-text" style="font-size: 2rem;">+</span><span class="blue-text" style="font-size: 2rem;">+</span>';
                                    content = '<span class="plus-split green-blue" style="font-size: 2rem;">+</span>';
                                }
                            } else if (isPathA || isPathB) {
                                // content = '+';
                                content = '<span class="' + (isPathA ? 'blue-text' : 'green-text') + '" style="font-size: 2rem;">+</span>';
                            }
                        }
                    } else {
                        // Default behavior for other previews (non-current trials)
                        if (isOverlap) {
                            if (i % 2 === 0) {
                                pathClass = 'half-half-blue-green';
                                // was: two pluses
                                // content = '<span class="blue-text" style="font-size: 2rem;">+</span><span class="green-text" style="font-size: 2rem;">+</span>';
                                content = '<span class="plus-split blue-green" style="font-size: 2rem;">+</span>';
                            } else {
                                pathClass = 'half-half-green-blue';
                                // content = '<span class="green-text" style="font-size: 2rem;">+</span><span class="blue-text" style="font-size: 2rem;">+</span>';
                                content = '<span class="plus-split green-blue" style="font-size: 2rem;">+</span>';
                            }
                        // ...existing code...
                        } else if (isPathA) {
                            pathClass = 'blue-path';
                            content = '+';
                        } else if (isPathB) {
                            pathClass = 'green-path';
                            content = '+';
                        }
                    }

                    if (previewIndex < currentTrialIndex || feedback) {
                        if (isStartA) {
                            upcomingHTML += `<div class="upcoming-cell-done ${observedClass} blue-path" id="${cellId}" data-row="${row}" data-col="${col}"  style="font-size: 1.0rem;" >`;
                            // upcomingHTML += bluePerson.outerHTML;
                            upcomingHTML += bluePerson;
                            upcomingHTML += `</div>`;
                        } else if (isStartB) {
                            upcomingHTML += `<div class="upcoming-cell-done ${observedClass} green-path" id="${cellId}" data-row="${row}" data-col="${col}" style="font-size: 1.0rem;" >`;
                            // upcomingHTML += greenPerson.outerHTML;
                            upcomingHTML += greenPerson;
                            upcomingHTML += `</div>`;
                        } else if (isGoalA) {
                            upcomingHTML += `<div class="upcoming-cell-done ${observedClass} blue-path" id="${cellId}" data-row="${row}" data-col="${col}" style="font-size: 1.0rem;" >🏠</div>`;
                        } else if (isGoalB) {
                            upcomingHTML += `<div class="upcoming-cell-done ${observedClass} green-path" id="${cellId}" data-row="${row}" data-col="${col}" style="font-size: 1.0rem;" >🏠</div>`;
                        } else if (isPathA || isPathB || isOverlap) {
                            upcomingHTML += `<div class="upcoming-cell-done ${observedClass} ${pathClass}" id="${cellId}" data-row="${row}" data-col="${col}" style="font-size: 1.5rem;">`;
                            upcomingHTML += content;
                            upcomingHTML += `</div>`;
                        } else {
                            upcomingHTML += `<div class="upcoming-cell-done ${observedClass}" id="${cellId}" data-row="${row}" data-col="${col}"></div>`;
                        }
                    } else {
                        if (isStartA) {
                            upcomingHTML += `<div class="upcoming-cell ${observedClass} ${pathClass}" id="${cellId}" data-row="${row}" data-col="${col}" style="font-size: 1.0rem;" >`;
                            // upcomingHTML += bluePerson.outerHTML;
                            upcomingHTML += bluePerson;
                            upcomingHTML += `</div>`;
                        } else if (isStartB) {
                            upcomingHTML += `<div class="upcoming-cell ${observedClass} ${pathClass}" id="${cellId}" data-row="${row}" data-col="${col}" style="font-size: 1.0rem;" >`;
                            // upcomingHTML += greenPerson.outerHTML;
                            upcomingHTML += greenPerson;
                            upcomingHTML += `</div>`;
                        } else if (isGoalA) {
                            upcomingHTML += `<div class="upcoming-cell ${observedClass} ${pathClass}" id="${cellId}" data-row="${row}" data-col="${col}" style="font-size: 1.0rem;" >🏠</div>`
                        } else if (isGoalB) {
                            upcomingHTML += `<div class="upcoming-cell ${observedClass} ${pathClass}" id="${cellId}" data-row="${row}" data-col="${col}" style="font-size: 1.0rem;" >🏠</div>`
                        } else if (isPathA || isPathB || isOverlap || isUpcomingPath) {

                            // just shadow for letters?
                            // if (keyAssignment && previewIndex === currentTrialIndex) {
                            //     upcomingHTML += `<div class="upcoming-cell ${observedClass} ${pathClass}" id="${cellId}" data-row="${row}" data-col="${col}" style="font-size: 1.5rem; 
                            //     -webkit-text-stroke: 0.5px black; text-shadow: 1px 1px 1px black;">`;
                            //     upcomingHTML += content;
                            //     upcomingHTML += `</div>`;
                            // } else {
                            //     upcomingHTML += `<div class="upcoming-cell ${observedClass} ${pathClass}" id="${cellId}" data-row="${row}" data-col="${col}" style="font-size: 1.5rem;">`;
                            //     upcomingHTML += content;
                            //     upcomingHTML += `</div>`;
                            // }
                            
                            // or shadow for letters and '+' symbol (need bigger font for '+')
                            if (keyAssignment && previewIndex === currentTrialIndex) {
                                upcomingHTML += `<div class="upcoming-cell ${observedClass} ${pathClass}" id="${cellId}" data-row="${row}" data-col="${col}" style="font-size: 1.5rem; 
                                -webkit-text-stroke: 0.5px black; text-shadow: 1px 1px 1px black;">`;
                                upcomingHTML += content;
                                upcomingHTML += `</div>`;
                            } else {
                                upcomingHTML += `<div class="upcoming-cell ${observedClass} ${pathClass}" id="${cellId}" data-row="${row}" data-col="${col}" style="font-size: 2rem; 
                                -webkit-text-stroke: 0.5px black; text-shadow: 1px 1px 1px black;">`;
                                upcomingHTML += content;
                                upcomingHTML += `</div>`;
                            }

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

// Resolve the color assigned to a key ('P' or 'Q').
function getColorForKey(assignments, key) {
    if (!assignments) return null;
    if (assignments[key] === 'blue' || assignments[key] === 'green') {
        return assignments[key]; // key -> color
    }
    if (assignments.blue === key) return 'blue';   // color -> key
    if (assignments.green === key) return 'green';
    return null;
}

// Declare global variables for practice grids and trial indices
let practice1Grid, practice2Grid, practice3Grid, practice4Grid, practice5Grid;
let practice1TrialIndex = 0, practice2TrialIndex = 0, practice3TrialIndex = 0, practice4TrialIndex = 0, practice5TrialIndex = 0; 
let currentTrialIndex = 0;
let totalCost = 0; // Keeps track of total cost across trials

// Function to initialize or reinitialize the practice grids
function initPractice() {
    currentTrialIndex = 0;
    totalCost = 0;
    practice1TrialIndex = 0;
    practice2TrialIndex = 0;
    practice3TrialIndex = 0;
    practice4TrialIndex = 0;
    practice5TrialIndex = 0;

    return Promise.all([
        loadPracticeGrid('assets/trial_sequences/expt_2/practice/expt_info/expt_2_info_1.json', 'practice1Grid').then(grid => practice1Grid = grid),
        loadPracticeGrid('assets/trial_sequences/expt_2/practice/expt_info/expt_2_info_2.json', 'practice2Grid').then(grid => practice2Grid = grid),
        loadPracticeGrid('assets/trial_sequences/expt_2/practice/expt_info/expt_2_info_3.json', 'practice3Grid').then(grid => practice3Grid = grid),
        loadPracticeGrid('assets/trial_sequences/expt_2/practice/expt_info/expt_2_info_4.json', 'practice4Grid').then(grid => practice4Grid = grid),
        loadPracticeGrid('assets/trial_sequences/expt_2/practice/expt_info/expt_2_info_5.json', 'practice5Grid').then(grid => practice5Grid = grid)
    ]).then(() => {
        console.log('All practice grids loaded successfully.');
    }).catch(error => {
        console.error('Error loading practice grids:', error);
        throw error;
    });
}

// Example usage: Call this function to initialize or reinitialize the grids
// initPractice().then(() => {
//     console.log('Practice grids are ready for use.');
// });



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

// 1. Add taxi character with animation
function createAvatar() {
    return `
        🚖
    `;
}

// 2. Add visual and audio feedback for costs
function animateAgent(path, binaryCosts, pauseAtEnd=false, callback) {
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
                        cellElement.style.backgroundColor = cost === -1 ? "rgb(203, 43, 43)" : "#b8b8d9"; // Red for toll, grey for free NEED TO MAKE SURE THESE MATCH HTML SHADES
                    }

                    if (cost === -1) {
                        trialCost++;

                        // Visual burst effect for toll cost
                        cellElement.innerHTML += '<div class="cost-burst">+$1 Toll</div>';
                        setTimeout(() => {
                            const burst = cellElement.querySelector('.cost-burst');
                            if (burst) burst.remove();
                        }, 500);

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
                        }, 500);
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
                mergeCosts(trialCost, callback, pauseAtEnd);
            }
        }
        
        // Start the animation sequence after a short delay
        setTimeout(step, 500);
    } else {
        // If there's no path, just merge costs and execute callback
        mergeCosts(null, callback, pauseAtEnd);
    }
}

// 4. Add animated transitions between trials
// 4. Add animated transitions between trials
function mergeCosts(trialCost, callback, pauseAtEnd=false) {
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
            
            trialCostElement.textContent = `You ran out of time! -$${trialCost}`;
            trialCostElement.style.color = "#f87171";
            trialCostElement.classList.remove("hidden");
            
            // setTimeout(() => {
            totalCostElement.style.color = originalColor;
            
            // Continue with normal animation flow after showing the message
            if (totalCostElement && trialCostElement) {
                // Add warning animation to cost display
                if (trialCost > 0) {
                    trialCostElement.classList.add("cost-animate");
                }
                
                // trialCostElement.style.transition = "transform 0.5s ease-in-out";
                // trialCostElement.style.transform = "translateY(-20px)";

                setTimeout(() => {
                totalCost += trialCost;
            
                const startCost = totalCost - trialCost;
                const duration = 1000;
                const startTime = performance.now();
            
                function animate(now) {
                    const elapsed = now - startTime;
                    const progress = Math.min(elapsed / duration, 1);
                    const currentCount = Math.floor(startCost + progress * trialCost);
                    totalCostElement.textContent = `-$${currentCount}`;
            
                    if (progress < 1) {
                        requestAnimationFrame(animate);
                    } else {
                        totalCostElement.textContent = `-$${totalCost}`;
                        // trialCostElement.textContent = `-$0`;
                        // trialCostElement.classList.remove("cost-animate");
                        trialCostElement.style.transform = "translateY(0)";
                    }
                }
            
                requestAnimationFrame(animate);
            }, 100);
            }
            // }, 500);
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
            
                const startCost = totalCost - trialCost;
                const duration = 1000;
                const startTime = performance.now();
            
                function animate(now) {
                    const elapsed = now - startTime;
                    const progress = Math.min(elapsed / duration, 1);
                    const currentCount = Math.floor(startCost + progress * trialCost);
                    totalCostElement.textContent = `-$${currentCount}`;
            
                    if (progress < 1) {
                        requestAnimationFrame(animate);
                    } else {
                        totalCostElement.textContent = `-$${totalCost}`;
                        // trialCostElement.textContent = `-$0`;
                        // trialCostElement.classList.remove("cost-animate");
                        trialCostElement.style.transform = "translateY(0)";
                    }
                }
            
                requestAnimationFrame(animate);
            }, 100);
        }
    }

    setTimeout(() => {
        const currentJob = document.querySelector(".grid-container");
        const upcomingJobs = document.querySelectorAll(".upcoming-job");
        if (upcomingJobs.length > 0 && !pauseAtEnd) {
            
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
        } else if (pauseAtEnd) {
            // Show the "Press spacebar to continue" text
            // document.getElementById("continue-text").style.display = "block";

            // also need to get rid of the line breaks
            // document.getElementById("line-break1").style.display = "none";
            // document.getElementById("line-break2").style.display = "none";
            // document.getElementById("line-break3").style.display = "none";
            // document.getElementById("line-break4").style.display = "none";

            // Enable spacebar input
            jsPsych.pluginAPI.getKeyboardResponse({
                callback_function: jsPsych.finishTrial, // Ends trial when spacebar is pressed
                valid_responses: [' '], // Spacebar
                rt_method: "performance",
                persist: false,
                allow_held_key: false
            });
            console.log('spacebar enabled');

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
    on_load: function() {
    },
    on_finish: function() {
    }
};

const pathPreSelectionTrial = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {
        // Randomly assign letters to blue and green paths
        const keyAssignment = Math.random() < 0.5 ? 
            // { blue: 'F', green: 'J' } : 
            // { blue: 'J', green: 'F' };
            { blue: 'Q', green: 'P' } : 
            { blue: 'P', green: 'Q' };
        
        // Store the assignment for this trial
        jsPsych.data.addProperties({
            blue_key: keyAssignment.blue,
            green_key: keyAssignment.green
        });
        return `
            <div class="jobs-layout">
            <div class="upcoming-jobs-container grid">
                ${grid.createAllJobsHTML(currentTrialIndex, null, keyAssignment)} 
            </div>
            </div>
        `;
    },
    choices: "NO_KEYS",
    trial_duration: 500, 
    on_finish: function() {
    }
};

// 5. Update the path selection trial to include taxi theme elements
const pathSelectionTrial = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {
        
        // Randomly assign letters to blue and green paths
        const keyAssignment = Math.random() < 0.5 ? 
            // { blue: 'F', green: 'J' } : 
            // { blue: 'J', green: 'F' };
            { blue: 'Q', green: 'P' } : 
            { blue: 'P', green: 'Q' };
        
        // Store the assignment for this trial
        jsPsych.data.addProperties({
            blue_key: keyAssignment.blue,
            green_key: keyAssignment.green
        });

        // or, get the key assignment from the last trial if we did preselection
        // const keyAssignment = {
        //     blue: jsPsych.data.get().last(1).values()[0].blue_key,
        //     green: jsPsych.data.get().last(1).values()[0].green_key
        // };
        
        return `
            <div class="jobs-layout">
                <div class="upcoming-jobs-container">
                    ${grid.createAllJobsHTML(currentTrialIndex, null, keyAssignment)} 
                </div>
            </div>
        `;
    },
    // choices: ['f', 'j'], 
    choices: ['q', 'p'], 
    trial_duration: 10000, // Automatically ends after 10 seconds
    on_finish: function(data) {
        // Get the key assignment for this trial
        const keyAssignment = {
            blue: jsPsych.data.get().last(1).values()[0].blue_key,
            green: jsPsych.data.get().last(1).values()[0].green_key
        };
        
        // Determine the chosen path based on the key pressed
        // let choice;
        // if (data.response === 'f') {
        //     choice = keyAssignment.blue === 'F' ? 'blue' : 'green';
        // } else if (data.response === 'j') {
        //     choice = keyAssignment.blue === 'J' ? 'blue' : 'green';
        // } else {
        //     choice = 'nan'; // Log as 'nan' if no response is made
        //     nTimeouts++;
        // }
        let choice;
        if (data.response === 'q') {
            choice = keyAssignment.blue === 'Q' ? 'blue' : 'green';
        } else if (data.response === 'p') {
            choice = keyAssignment.blue === 'P' ? 'blue' : 'green';
        } else {
            choice = 'nan'; // Log as 'nan' if no response is made
            nTimeouts++;
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
            animateAgent(chosenPath, binaryCosts, false, function() {
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
    body.style.backgroundImage = `url('assets/cities/cropped2/${mappedCityId}.png')`;
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
    oldCity.style.backgroundImage = `url('assets/cities/cropped2/${oldCityMapping}.png')`;
    oldCity.style.backgroundSize = 'cover';
    oldCity.style.backgroundPosition = 'center';
    transitionContainer.appendChild(oldCity);
    
    // Create new city element
    let newCity = document.createElement('div');
    let newCityMapping;
    newCity.style.width = '50%'; // Half of the container
    newCity.style.height = '100%';
    newCityMapping = cityMapping[newCityId];
    newCity.style.backgroundImage = `url('assets/cities/cropped2/${newCityMapping}.png')`;
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
                <h3>You paid a total of <strong style="color:rgb(203, 43, 43);">$${todayTolls}</strong> in tolls today.</h3>
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
                <h3>You would have paid a total of <strong style="color:  rgb(203, 43, 43);">$${todayTolls}</strong> in tolls today.</h3>
                <h3>Press spacebar to continue.</h3>
            </div>
        `;
    },
choices: [' '], // spacebar to continue
on_load: function() {
},
on_finish: function() {
}
};

const timeoutCheck = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {
        // Get the current city ID and the previous city ID
        const previousCityId = grid.getCurrentCity();
        console.log(`Number of timeouts in city ${previousCityId}:`, nTimeouts);

        // Check if the number of timeouts exceeds the threshold
        const nTrials = grid.nTrials;
        const nGrids = grid.nGrids;
        const nTrialsPerCity = nTrials * nGrids;
        const threshold = Math.floor(0.3 * nTrialsPerCity);

        if (nTimeouts > threshold) {
            console.log(`Participant failed due to high number of timeouts in city ${previousCityId}`);
            return `
                <div class="instruction-section">
                    <h2>Experiment Failed</h2>
                    <p>You have timed out too many times in the previous city. Unfortunately, you cannot continue with the experiment.</p>
                    <p>Please wait while you are redirected to Prolific. You will be paid for your time.</p>
                    <p style="font-weight: bold; ">DO NOT CLOSE YOUR BROWSER until you have returned to Prolific. This may take a few seconds...</p>
                </div>
            `;
        } else {
            // If successful, reset nTimeouts and proceed
            nTimeouts = 0;
            return null;
        }
    },
    choices: "NO_KEYS", // Disable keypresses
    on_load: function() {
        // If the participant passed the check, immediately finish the trial
        if (nTimeouts <= Math.floor(0.3 * grid.nTrials * grid.nGrids)) {
            jsPsych.finishTrial();
        } else {
            
            // If the participant failed, send the data and redirect
            // const ppt_data = jsPsych.data.get().json();

            // Define the variables you want to keep
            // var keep_vars = [
            //     "pid", "trial", "city", "path_chosen", "button_pressed", "reaction_time_ms","html-keyboard-response","choice",
            //     "context", "grid", "path_A_expected_cost", "path_B_expected_cost",
            //     "path_A_actual_cost", "path_B_actual_cost", "path_A", "path_B",
            //     "path_A_future_overlap", "path_B_future_overlap",
            //     "path_A_future_row_overlap", "path_B_future_row_overlap",
            //     "path_A_future_col_overlap", "path_B_future_col_overlap",
            //     "path_A_future_row_and_col_overlap", "path_B_future_row_and_col_overlap",
            //     "path_A_future_rel_overlap", "path_B_future_rel_overlap",
            //     "path_A_future_irrel_overlap", "path_B_future_irrel_overlap",
            //     "abstract_sequence_A", "abstract_sequence_B",
            //     "dominant_axis_A", "dominant_axis_B",
            //     "better_path", "chose_better_path", "bonusAchieved", "expt_info_filename"
            // ];

            // Filter the jsPsych data down to just those variables
            // var ppt_data = jsPsych.data.get().filterColumns(keep_vars).json();
            // console.log('experiment complete (filtered dataset)');
            // var ppt_data = jsPsych.data.get()
            //     .filter({ trial_type: 'html-keyboard-response' }) // only keep choice trials
            //     .json();
            // console.log('experiment complete (filtered dataset)');
            
            var ppt_data = JSON.stringify(
                jsPsych.data.get()
                .filter({ trial_type: 'html-keyboard-response' })
                .values()
                .map(({ stimulus, ...rest }) => rest)
            );
            console.log('experiment complete (filtered dataset)');
            
            
            send_complete(subject_id, ppt_data)
                .then(() => {
                    console.log('Data successfully sent to completion endpoint');
                    setTimeout(() => {
                        window.location.replace("https://app.prolific.com/submissions/complete?cc=C12CZWYW");
                    }, 2000); // Redirect after 3.5 seconds
                })
                .catch(error => {
                    console.error('Failed to send completion data:', error);
                    // Add a note about the error but still redirect after a delay
                    document.querySelector('.instruction-section').innerHTML += `
                        <p>Note: There might have been an issue saving your data, but you will still be redirected shortly.</p>
                    `;
                    setTimeout(() => {
                        window.location.replace("https://app.prolific.com/submissions/complete?cc=C12CZWYW");
                    }, 2000);
                });
        }
    }
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
                <h2>City ${currentCityId}/${nCities} complete.</h2>
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

let nTimeouts = 0;
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
    on_load: function() {
    },
    on_finish: function() {
        console.log("Experiment has begun in City:", grid.getCurrentCity()), ', Trial:', currentTrialIndex,', Grid:', grid.getTrialInfo(currentTrialIndex).grid;
        var ppt_data = jsPsych.data.get().json();
        send_incomplete(subject_id, ppt_data);
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
                <p>Before we check if you received your bonus, we have a few questions that we would like to ask you about your experience of the task - we would love to hear your thoughts.</p>
                <p>Press spacebar to continue.</p>
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
    on_finish: function(data) {
        data.bonusAchieved = bonusAchieved;
        var ppt_data = jsPsych.data.get().json();
        send_incomplete(subject_id, ppt_data);
    }
};

const preQuestionnaire = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {
        return `
            <div class="new-day-text">
                <h2>Final survey</h2>
                <p>We would now like to ask you a few survey questions. Press respond to every question you feel comfortable answering.</p>
                <p>Once you are done, you will find out if you received your bonus, and you will then be returned to Prolific.</p>
                <p>Press spacebar to continue.</p>
            </div>
        `;
    },
choices: [' '], // spacebar to continue
    on_finish: function(data) {
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

        // appropriate zooming
        document.body.style.overflowY = "hidden";
        document.documentElement.style.overflowY = "hidden";
        applyScreenScaling();

        // Set initial city background to 'practice1.png'
        setCityBackground('practice1');
        grid.currentCity = 'practice1'; // Initialize the current city
    },
    on_finish: function() {
    }
};

const instructions2 = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {
        return `
            <div class="instruction-section" style="font-size: 20px;">
                <h1>Dispatch Instructions:</h1>
                <p>For each dispatch, you'll see two possible jobs marked in <span class="blue-text">blue</span> and <span class="green-text">green</span>. Each job has a passenger ✋ at a pickup point, and a drop-off destination 🏠. The route of each job is marked with one of two letters:</p>
                <p>- The letter <strong>P</strong> marks one job</p>
                <p>- The letter <strong>Q</strong> marks the other job</p>
                <p>On each dispatch, these letters are randomly assigned to each job. To send out a taxi to one of these jobs, you need to press the corresponding key on your keyboard. Note that if an intersection appears on both paths, it will contain both P and Q.</p>
                <p>For any given choice, the lengths of the two possible jobs are the same, and you are paid the same wage by the company each day. However, some jobs are more costly than others, which you must pay for yourself. This is because of tolls in the city...</p>
            </div>
            <div class="instruction-section" style="font-size: 20px;">
                <h2>Press spacebar to continue.</h2>
            </div>
        `;
    },
    choices: [' '], // Spacebar to continue
    on_load: function() {
        initPractice(); // Initialize the grid for practice1
        console.log('pratice1TrialIndex', practice1TrialIndex)
    },
    on_finish: function(data) {
    }
    
};

const instructions2_5 = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {
        const fontSize = "20px"; // Define font size as a variable
        return `
            <div class="instruction-section" style="font-size: 20px;">
                <h1>Toll Intersections:</h1>
                <p>Traffic in some parts of the city is busier than in others. This means that tolls apply at busy intersections. Visiting an intersection reveals whether or not you have to pay a toll there.</p>
                <p>- <strong><span style="color: rgb(114, 114, 150);">Dark grey intersections</span></strong> have not been visited yet</p>
                <p>- <strong><span style="color: rgb(203, 43, 43);">Red intersections</span></strong> cost a $1 toll to pass through</p>
                <p>- <strong><span style="color:rgb(194, 194, 229);">Light grey intersections</span></strong> are free with no tolls</p>
                <p>Your goal is to complete all taxi jobs while minimising the total tolls paid out.</p>
                <p>Note that the rows and columns of the city are labelled with numbers and letters, respectively, to improve readability.</p>
            </div>

            <div class="instruction-section" style="font-size: 20px;">
                <h2>Press spacebar to practise selecting a job.</h2>
            </div>
        `;
    },
    choices: [' '], // Spacebar to continue
    on_load: function() {
    },
    on_finish: function(data) {
    }
    
};

const practice1SelectionTrial = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {
        

        // Determine the key assignment based on the trial index
        // const keyAssignment = { blue: 'F', green: 'J' };
        const keyAssignment = { blue: 'Q', green: 'P' };
        const instruction = practice1TrialIndex === 0 
            ? `<h3>Please select the <span style="color: rgb(47, 164, 253); font-weight: bold;">BLUE</span> path by pressing the <span style="font-weight: bold;">${keyAssignment.blue}</span> key.</h3>`
            : `<h3>Please select the <span style="color:  #29da6a; font-weight: bold;">GREEN</span> path by pressing the <span style="font-weight: bold;">${keyAssignment.green}</span> key.</h3>`;
        
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
        // return practice1TrialIndex === 0 ? ['f'] : ['j'];
        return practice1TrialIndex === 0 ? ['q'] : ['p'];
    },
    on_load: function() {
    },
    on_finish: function(data) {
        
        // Get the key assignment for this trial
        const keyAssignment = {
            blue: jsPsych.data.get().last(1).values()[0].blue_key,
            green: jsPsych.data.get().last(1).values()[0].green_key
        };
        
        // Determine the choice based on the key pressed
        // let choice;
        // if (data.response === 'f') {
        //     choice = keyAssignment.blue === 'F' ? 'blue' : 'green';
        // } else if (data.response === 'j') {
        //     choice = keyAssignment.green === 'J' ? 'green' : 'blue';
        // }
        let choice;
        if (data.response === 'q') {
            choice = keyAssignment.blue === 'Q' ? 'blue' : 'green';
        } else if (data.response === 'p') {
            choice = keyAssignment.green === 'P' ? 'green' : 'blue';
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
            animateAgent(chosenPath, binaryCosts, false, function() {
                jsPsych.finishTrial();
            });
        }, 100);
    }
};

const instructions3_1 = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {
        const n = grid.nTrials;
        const fontSize = "22px"; // Define font size as a variable
        const selectedPath = null;
        const keyAssignment = { blue: 'Q', green: 'P' }
        jsPsych.data.addProperties({
            blue_key: keyAssignment.blue,
            green_key: keyAssignment.green
        });
        const feedback=false;
        const firstDay=false;
        const showPink=false;
        return `
        <div class="cost-display-container">
            <h1>Daily Shift:</h1>
            <p style="font-size: ${fontSize};">Each day, you will manage ${n} dispatches, meaning you have ${n} jobs to select. All ${n} pairs of jobs will be presented on screen at once, side-by-side.</p>
            <p style="font-size: ${fontSize};">Each dispatch takes place at a different time of the day and is marked with one of the following clock icons, displayed above the dispatch:</p>
            <p style="font-family: golemClocks; text-align: center; font-size: ${fontSize};">&#x00E6; &#x00DD; &#x0026; &#x263A;</p>
            <p style="font-size: ${fontSize};">You will move through these dispatches from the left- to the right-hand side of the screen. The clock above your current dispatch is highlighted in <span style="color: #ece75d;">yellow</span>.</p>
            <h2 style="font-size: ${fontSize};">Press spacebar to continue.</h2>
        </div>
        <div class="jobs-layout">
            <div class="upcoming-jobs-container grid">
                ${practice2Grid.createAllJobsHTML(practice2TrialIndex, selectedPath, keyAssignment, feedback, firstDay,showPink).replace(/<div id="cost-message".*?<\/div>/s, '')} 
            </div>
        </div>
        `;
    },
    choices: [' '], // Spacebar to continue
    on_load: function() {
    },
    on_finish: function() {
    }
};

const instructions3_2 = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {
        const n = grid.nTrials;
        const fontSize = "22px"; // Define font size as a variable
        const selectedPath = null;
        const keyAssignment = null;
        const feedback=false;
        const firstDay=false;
        const showPink=false;
        return `
        <div class="cost-display-container">
            <h1>Daily Shift:</h1>
            <p style="font-size: ${fontSize};">All ${n} pairs of jobs will be presented on screen at once, side-by-side.</p>
            <p style="font-size: ${fontSize};">This means you will also be able to see details about your upcoming dispatches - that is, you will be able to see the jobs that you will have to choose between later in the day.</p>
            <p style="font-size: ${fontSize};">See below how your upcoming dispatches are displayed on screen to the right of your current dispatch.</p>
            <br>
            <br>
            <br>
            <br>
            <h2 style="font-size: ${fontSize};">Press spacebar to continue.</h2>
        </div>
        <div class="jobs-layout">
            <div class="upcoming-jobs-container grid">
            ${practice2Grid.createAllJobsHTML(practice2TrialIndex, selectedPath, keyAssignment, feedback, firstDay, showPink).replace(/<div id="cost-message".*?<\/div>/s, '')} 
            </div>
        </div>
        `;
    },
    choices: [' '], // Spacebar to continue
    on_load: function() {
    },
    on_finish: function() {
    }
};

const instructions3_3_1 = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {
        const n = grid.nTrials;
        const fontSize = "22px"; // Define font size as a variable
        const selectedPath = null;
        const keyAssignment = { blue: 'Q', green: 'P' }
        jsPsych.data.addProperties({
            blue_key: keyAssignment.blue,
            green_key: keyAssignment.green
        });
        const feedback=false;
        const firstDay=false;
        const showPink=true;
        const restrictPink=1;
        return `
        <div class="cost-display-container">
            <h1>Daily Shift:</h1>
            <p style="font-size: ${fontSize};">As well as being shown individually, information about your upcoming dispatches will also be highlighted in your <span style="color: #ece75d;">current dispatch</span>.</p>
            <p style="font-size: ${fontSize};">Specifically, the intersections that you may possibly visit later in the day are highlighted in <span style="color: rgb(240, 110, 254);">pink</span>.</p>
            <p style="font-size: ${fontSize};">See below how the intersections that may be visited in your <strong>second</strong> dispatch are also displayed in your <span style="color: #ece75d;">current dispatch</span>.</p>
            <br>
            <br>
            <br>
            <br>
            <h2 style="font-size: ${fontSize};">Press spacebar to continue.</h2>
        </div>
        <div class="jobs-layout">
            <div class="upcoming-jobs-container grid">
            ${practice2Grid.createAllJobsHTML(practice2TrialIndex, selectedPath, keyAssignment, feedback, firstDay, showPink, restrictPink).replace(/<div id="cost-message".*?<\/div>/s, '')} 
            </div>
        `;
    },
    choices: [' '], // Spacebar to continue
    on_load: function() {
    },
    on_finish: function() {
    }
};
const instructions3_3_2 = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {
        const n = grid.nTrials;
        const fontSize = "22px"; // Define font size as a variable
        const selectedPath = null;
        const keyAssignment = { blue: 'Q', green: 'P' }
        jsPsych.data.addProperties({
            blue_key: keyAssignment.blue,
            green_key: keyAssignment.green
        });
        const feedback=false;
        const firstDay=false;
        const showPink=true;
        const restrictPink=2;
        return `
        <div class="cost-display-container">
            <h1>Daily Shift:</h1>
            <p style="font-size: ${fontSize};">As well as being shown individually, information about your upcoming dispatches will also be highlighted in your <span style="color: #ece75d;">current dispatch</span>.</p>
            <p style="font-size: ${fontSize};">Specifically, the intersections that you may possibly visit later in the day are highlighted in <span style="color: rgb(240, 110, 254);">pink</span>.</p>
            <p style="font-size: ${fontSize};">See below how the intersections that may be visited in your <strong>second or third</strong> dispatch are also displayed in your <span style="color: #ece75d;">current dispatch</span>.</p>
            <br>
            <br>
            <br>
            <br>
            <h2 style="font-size: ${fontSize};">Press spacebar to continue.</h2>
        </div>
        <div class="jobs-layout">
            <div class="upcoming-jobs-container grid">
            ${practice2Grid.createAllJobsHTML(practice2TrialIndex, selectedPath, keyAssignment, feedback, firstDay, showPink, restrictPink).replace(/<div id="cost-message".*?<\/div>/s, '')} 
            </div>
        `;
    },
    choices: [' '], // Spacebar to continue
    on_load: function() {
    },
    on_finish: function() {
    }
};
const instructions3_3_3 = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {
        const n = grid.nTrials;
        const fontSize = "22px"; // Define font size as a variable
        const selectedPath = null;
        const keyAssignment = { blue: 'Q', green: 'P' }
        jsPsych.data.addProperties({
            blue_key: keyAssignment.blue,
            green_key: keyAssignment.green
        });
        const feedback=false;
        const firstDay=false;
        const showPink=true;
        return `
        <div class="cost-display-container">
            <h1>Daily Shift:</h1>
            <p style="font-size: ${fontSize};">As well as being shown individually, information about your upcoming dispatches will also be highlighted in your <span style="color: #ece75d;">current dispatch</span>.</p>
            <p style="font-size: ${fontSize};">Specifically, the intersections that you may possibly visit later in the day are highlighted in <span style="color: rgb(240, 110, 254);">pink</span>.</p>
            <p style="font-size: ${fontSize};">See below how the intersections that may be visited in <strong>any</strong> of your upcoming dispatches are also displayed in your <span style="color: #ece75d;">current dispatch</span>.</p>
            <br>
            <br>
            <br>
            <br>
            <h2 style="font-size: ${fontSize};">Press spacebar to continue.</h2>
        </div>
        <div class="jobs-layout">
            <div class="upcoming-jobs-container grid">
            ${practice2Grid.createAllJobsHTML(practice2TrialIndex, selectedPath, keyAssignment, feedback, firstDay, showPink).replace(/<div id="cost-message".*?<\/div>/s, '')} 
            </div>
        `;
    },
    choices: [' '], // Spacebar to continue
    on_load: function() {
    },
    on_finish: function() {
    }
};

const instructions3_4 = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {
        const n = grid.nTrials;
        const fontSize = "22px"; // Define font size as a variable
        const selectedPath = null;
        const keyAssignment = { blue: 'Q', green: 'P' }
        jsPsych.data.addProperties({
            blue_key: keyAssignment.blue,
            green_key: keyAssignment.green
        });
        const feedback=false;
        const firstDay=false;
        const showPink=2;
        return `
        <div class="cost-display-container">
            <h1>Daily Shift:</h1>
            <p style="font-size: ${fontSize};">You can select your desired job once the clock above your current dispatch turns <span style="color: #ece75d;">yellow</span>.</p>
            <p style="font-size: ${fontSize};">You will have 10 seconds to select a job by pressing either P or Q. If you fail to make a choice within this time limit, you will pay a fine of <span style="color: #f87171;">$10</span>.</p>
            <br>
            <br>
            <br>
            <br>
            <br>
            <br>
            <h2 style="font-size: ${fontSize};">Press P or Q to select one of the jobs.</h2>
        </div>
        <div class="jobs-layout">
            <div class="upcoming-jobs-container grid">
            ${practice2Grid.createAllJobsHTML(practice2TrialIndex, selectedPath, keyAssignment, feedback, firstDay, showPink).replace(/<div id="cost-message".*?<\/div>/s, '')} 
            </div>
        </div>
        `;
    },
    // choices: [' '], // Spacebar to continue
    choices: ['q', 'p'], 
    on_load: function() {
    },
    on_finish: function(data) {
        
        // Get the key assignment for this trial
        const keyAssignment = {
            blue: jsPsych.data.get().last(1).values()[0].blue_key,
            green: jsPsych.data.get().last(1).values()[0].green_key
        };
        
        // Determine the chosen path based on the key pressed
        let choice;
        if (data.response === 'q') {
            choice = keyAssignment.blue === 'Q' ? 'blue' : 'green';
        } else if (data.response === 'p') {
            choice = keyAssignment.blue === 'P' ? 'blue' : 'green';
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
        console.log('keyAssignment', keyAssignment);
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

const instructions3_5 = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {
        
        const lastTrialData = jsPsych.data.get().last(1).values()[0];
        const keyAssignment = lastTrialData.key_assignment;
        const selectedPath = lastTrialData.choice;
        console.log('selectedPath', selectedPath, 'keyAssignment', keyAssignment);

        const fontSize = "22px"; // Define font size as a variable
        return `
        <div class="cost-display-container">
            <h1>Daily Shift:</h1>
            <p style="font-size: ${fontSize};">You can select your desired job once the clock above your current dispatch turns <span style="color: #ece75d;">yellow</span>.</p>
            <p style="font-size: ${fontSize};">You will have 10 seconds to select a job by pressing either P or Q. If you fail to make a choice within this time limit, you will pay a fine of <span style="color: #f87171;">$10</span>.</p>
            <br>
            <br>
            <br>
            <br>
            <br>
            <br>
            <h2 style="font-size: ${fontSize};">Once the job is complete, press spacebar to continue.</h2>
        </div>
        <div class="jobs-layout">
            <div class="upcoming-jobs-container grid">
            ${practice2Grid.createAllJobsHTML(practice2TrialIndex, selectedPath, keyAssignment).replace(/<div id="cost-message".*?<\/div>/s, '')} 
            </div>
        </div>
        `;
    },
    choices: "NO_KEYS", // initially disable keypress
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
        

        const pauseAtEnd = true;
        animateAgent(chosenPath, binaryCosts, pauseAtEnd, function() {

            jsPsych.finishTrial();

        });
        
        // Increment the practice trial index
        // practice2TrialIndex++;

        // After animation completes, enable spacebar
        // setTimeout(() => {

        //     // Show the "Press spacebar to continue" text
        //     document.getElementById("continue-text").style.display = "block";

        //     // Enable spacebar input
        //     jsPsych.pluginAPI.getKeyboardResponse({
        //         callback_function: jsPsych.finishTrial, // Ends trial when spacebar is pressed
        //         valid_responses: [' '], // Spacebar
        //         rt_method: "performance",
        //         persist: false,
        //         allow_held_key: false
        //     });
        // }, 1000);
    },
    on_finish: function() {
        
        // Increment the practice trial index
        practice2TrialIndex++;
        console.log('incremented practice2TrialIndex to', practice2TrialIndex);
    }
};

const instructions3_6 = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {
        const n = grid.nTrials;
        const fontSize = "22px"; // Define font size as a variable
        const selectedPath = null;
        const keyAssignment = Math.random() < 0.5 ? 
            { blue: 'Q', green: 'P' } : 
            { blue: 'P', green: 'Q' };
        jsPsych.data.addProperties({
            blue_key: keyAssignment.blue,
            green_key: keyAssignment.green
        });
        const feedback=false;
        const firstDay=false;
        const showPink=n;
        return `
        <div class="cost-display-container">
            <h1>Daily Shift:</h1>
            <p style="font-size: ${fontSize};">As you move through the day’s dispatches from left to right, past dispatches are <span style="color: rgb(138, 138, 184);">greyed out</span>.</p>
            <p style="font-size: ${fontSize};">The locations of tolls remain fixed throughout the day. Once you visit an intersection, you find out whether or not you have to pay a toll whenever you reach that intersection again on the same day.</p>
            <p style="font-size: ${fontSize};">Notice how, whenever you visit an intersection, information about whether it contains a toll or not also becomes available in your upcoming dispatches.</p>
            <p style="font-size: ${fontSize};">Hence, finding out about the intersections will help you for the rest of the day, since it allows you to select jobs where you don’t have to pay any tolls.</p>
            <h2 style="font-size: ${fontSize};">Press spacebar to continue.</h2>
        </div>
        <div class="jobs-layout">
            <div class="upcoming-jobs-container grid">
            ${practice2Grid.createAllJobsHTML(practice2TrialIndex, selectedPath, keyAssignment, feedback, firstDay, showPink).replace(/<div id="cost-message".*?<\/div>/s, '')} 
            </div>
        </div>
        `;
    },
    choices: [' '], // Spacebar to continue
    on_load: function() {
    },
    on_finish: function() {

    }
};

const instructions3_7 = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {
        const fontSize = "20px"; // Define font size as a variable
        return `
            <div class="instruction-section" style="font-size: 20px;">
                <h1>Practice Shift:</h1>
                <p>You will now practise a full day of dispatches.</p>
                <p>The total amount of tolls paid over the course of the day will be shown at the top of your screen.</p>
                <p>Before this practice, you have the opportunity to review the most recent instructions.</p>
                <p>Press backspace to review, or spacebar to continue.</p>
            </div>
        `;
    },
    choices: [' ', 'backspace'], // allow both
    on_load: function() {
    },
    on_finish: function(data) {
    data.restart_instructions = (data.response === 'backspace');

    // if restarting instructions, need to clear practice2TrialIndex and practice2Grid
    if (data.restart_instructions) {
        // practice2TrialIndex = 0;
        // loadPracticeGrid('assets/trial_sequences/expt_2/practice/expt_info/expt_2_info_2.json', 'practice2Grid').then(grid => practice2Grid = grid);
        initPractice(); // Initialize the grid for practice1

        //check that grid has no observed costs in it
        for (let i = 0; i < practice2Grid.nTrials; i++) {
            practice2Grid[`observedCosts${i}`] = {};
        }
    }
}   
};

const instructions3_node = {
  timeline: [
    instructions3_1,
    // instructions3_2,
    instructions3_3_1,
    instructions3_3_2,
    instructions3_3_3,
    instructions3_4,
    instructions3_5,
    instructions3_6,
    instructions3_7,
  ],
  loop_function: function(data) {
    const last = data.values().slice(-1)[0];      
    return !!(last.restart_instructions || last.response === 'backspace');
  }
};

const practiceFirstDayTrial = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {
        // document.body.style.zoom = zoomFactor;
        return `
            <div class="jobs-layout">
            <div class="upcoming-jobs-container grid ${practice3TrialIndex % practice3Grid.nTrials === 0 ? 'grid-fade-in' : ''}">
                ${practice3Grid.createAllJobsHTML(practice3TrialIndex, null, null, false, true)} 
            </div>
            </div>
        `;
    },
    choices: "NO_KEYS",
    trial_duration: 3000, // 
    on_load: function() {
    },
    on_finish: function() {
    }
};

const practice3PreSelectionTrial = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {
        // Reset total cost for practice if first practice trial
        if (practice3TrialIndex === 0) {
            totalCost = 0;
        }

        // Randomly assign letters to blue and green paths
        const keyAssignment = Math.random() < 0.5 ? 
            { blue: 'Q', green: 'P' } : 
            { blue: 'P', green: 'Q' };
        
        // Store the assignment for this trial
        jsPsych.data.addProperties({
            blue_key: keyAssignment.blue,
            green_key: keyAssignment.green
        });
        return `
            <div class="jobs-layout">
            <div class="upcoming-jobs-container grid">
                ${practice3Grid.createAllJobsHTML(practice3TrialIndex, null, keyAssignment)} 
            </div>
            </div>
        `;
    },
    choices: "NO_KEYS",
    trial_duration: 500, // Ends after 500ms
    on_finish: function() {
    }
};

const practice3SelectionTrial = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {

        // randomly assign keys
        const keyAssignment = Math.random() < 0.5 ? 
            { blue: 'Q', green: 'P' } : 
            { blue: 'P', green: 'Q' };
        
        // Store the assignment for this trial
        jsPsych.data.addProperties({
            blue_key: keyAssignment.blue,
            green_key: keyAssignment.green
        });

        // or get the key assignment for this trial if we did preselection trial
        // const keyAssignment = {
        //     blue: jsPsych.data.get().last(1).values()[0].blue_key,
        //     green: jsPsych.data.get().last(1).values()[0].green_key
        // };

        // Reset total cost for practice if first practice trial
        if (practice3TrialIndex === 0) {
            totalCost = 0;
        }
        const totalCostElement = document.getElementById("total-cost");
        if (totalCostElement) {
            totalCostElement.textContent = "$0";
        }
        
        return `
            <div class="jobs-layout">
                <div class="upcoming-jobs-container">
                    ${practice3Grid.createAllJobsHTML(practice3TrialIndex, null, keyAssignment)} 
                </div>
            </div>
        `;
    },
    // choices: ['f', 'j'], 
    choices: ['q', 'p'], 
    trial_duration: 10000, // Automatically ends after 10 seconds
    on_finish: function(data) {
        // Get the key assignment for this trial
        const keyAssignment = {
            blue: jsPsych.data.get().last(1).values()[0].blue_key,
            green: jsPsych.data.get().last(1).values()[0].green_key
        };
        
        // Determine the chosen path based on the key pressed
        let choice;
        if (data.response === 'q') {
            choice = keyAssignment.blue === 'Q' ? 'blue' : 'green';
        } else if (data.response === 'p') {
            choice = keyAssignment.blue === 'P' ? 'blue' : 'green';
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
            gridContainer.innerHTML = practice3Grid.createGridHTML(practice3TrialIndex, choice, keyAssignment,true,true);
        }
        
        // // Store all the relevant data from the current trial
        const currentTrial = practice3Grid.getTrialInfo(practice3TrialIndex);
        data.choice = choice;
        data.trial = currentTrial.trial;
        data.city = currentTrial.city;
        data.grid_id = currentTrial.practice3Grid;
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

const practice3AnimationTrial = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {
        const lastTrialData = jsPsych.data.get().last(1).values()[0];
        const keyAssignment = lastTrialData.key_assignment;
        
        return `
            <div class="jobs-layout">
                <div class="upcoming-jobs-container">
                    ${practice3Grid.createAllJobsHTML(practice3TrialIndex, lastTrialData.choice, keyAssignment)} 
                </div>
            </div>
        `;
    },
    choices: "NO_KEYS",
    on_load: function() {
        const currentTrial = practice3Grid.getTrialInfo(practice3TrialIndex);
        const lastTrialData = jsPsych.data.get().last(1).values()[0];

        // hacky
        currentTrialIndex = practice3TrialIndex;

        if (!lastTrialData || !lastTrialData.choice) {
            console.error("No valid path choice found. Restarting trial.");
            return jsPsych.finishTrial();
        }

        const chosenPath = lastTrialData.choice === 'blue' ? currentTrial.path_A : 
                   lastTrialData.choice === 'green' ? currentTrial.path_B : 
                   null;
        const binaryCosts = practice3Grid.getBinaryCosts(`city_${currentTrial.city}_grid_${currentTrial.grid}`);

        if (chosenPath !== null) {
            practice3Grid.recordObservedCosts(chosenPath, binaryCosts);
        }
        

        setTimeout(() => {
            animateAgent(chosenPath, binaryCosts, false, function() {
                jsPsych.finishTrial();
            });
        }, 100);
        
        // Increment the practice trial index
        practice3TrialIndex++;
    }
};

// show how the grid resets at the end of the day
const instructions4 = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: `
        <div class="instruction-section" style="z-index: 2001; position: relative;">
            <h1>New Day</h1>
            <p>At the start of a new day, the traffic in the city resets, meaning that the intersections where you do (or do not) have to pay a toll have reset. You are also given a new set of dispatches.</p>
            <p>Watch the grid reset for the next day below.</p>
            <p id="continue-text" style="display: none;">Press spacebar to continue dispatching.</p>
        </div>
        <div class="jobs-layout" style="z-index: 2001; position: relative;">
            <div id="grid-container" class="current-job-section"></div>
        </div>
    `,
    // <p>Remember: it helps to think about which intersections you might visit later on in the day, as highlighted in <span style="color: rgb(240, 110, 254);">pink</span>, and shown in your upcoming dispatches.</p>
    choices: "NO_KEYS", // Initially disable keypresses
    on_load: function() {

        // Show the current state of the grid for 1 second
        const gridContainer = document.getElementById('grid-container');
        // gridContainer.innerHTML = practice3Grid.createGridHTML(practice3TrialIndex, 'none', null, false); // Render the current state of the grid without cost display, nor any current paths
        const revealCosts = true; // Set to true to show costs
        gridContainer.innerHTML = practice3Grid.createBlankGridHTML(practice3TrialIndex, revealCosts, false, 'observed'); // Render a blank grid

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
            gridContainer.innerHTML = practice3Grid.createBlankGridHTML(); // Render a blank grid
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
        practice3Grid.resetGrid(); // Reset the grid for the new set of trials
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
        oldCity.style.backgroundImage = `url('assets/cities/cropped2/practice1.png')`;
        oldCity.style.backgroundSize = 'cover';
        oldCity.style.backgroundPosition = 'center';
        transitionContainer.appendChild(oldCity);
        
        // Create new city element
        let newCity = document.createElement('div');
        newCity.style.width = '50%'; // Half of the container
        newCity.style.height = '100%';
        newCity.style.backgroundImage = `url('assets/cities/cropped2/practice2.png')`;
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
    },
    on_finish: function() {
    }
};
    


// illustrate column city
const instructions6 = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {
        const fontSize = "28px"; // Define font size as a variable
        const zoomTmp = zoomFactor * 0.75
        document.body.style.zoom = zoomTmp;
        // document.body.style.zoom = "75%";
        return `
            <div class="instruction-section" style="margin: 10px;">
            <h1>How can you predict which intersections have tolls (or no tolls)?</h1>
            <p style="font-size: ${fontSize};">Each city has particular traffic properties, such that the busy streets tend to be related to one another in one of two ways.</p>
            </div>
            <div class="instruction-section" style="margin: 10px;">
            <h1>'Column cities'</h1>
            <p style="font-size: ${fontSize};">In column cities, traffic tends to run from north to south every day, meaning that tolls tend to be clustered in columns.</p>
            <p style="font-size: ${fontSize};">That is, a column may have a lot of tolls, or not many tolls.</p>
            <p style="font-size: ${fontSize};">The particular locations of these busy columns may change each day, but the city will always have this column-dependent feature.</p>
            <p style="font-size: ${fontSize};">Press spacebar to continue.</p>
            </div>
            <div class="jobs-layout" >
            <div class="instruction-section" style="text-align: center; font-size: 20px; color: #3a3a3a; margin: 10px;">
            <h3><strong>Example day ${practice4TrialIndex + 1} tolls</strong><h3>
            </div>
            <div id="grid-container" class="current-job-section"></div>
        `;
    },
    choices: [' '], // Wait for spacebar to continue
    on_load: function() {

        const gridContainer = document.getElementById('grid-container');
        const revealCosts = true; // Set to true to show costs
        console.log("Rendering grid for practice4TrialIndex:", practice4TrialIndex);
        gridContainer.innerHTML = practice4Grid.createBlankGridHTML(practice4TrialIndex, revealCosts); // Render a blank grid
        practice4TrialIndex++;
    },
    on_finish: function() {
    }
};

// illustrate city change
const instructions7 = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {
        const fontSize = "28px"; // Define font size as a variable
        return `
            <div class="instruction-section" style="z-index: 2000; position: relative;">
            <h1>How can you predict which intersections have tolls (or no tolls)?</h1>
            <p style="font-size: ${fontSize};">Each city has particular traffic properties, such that the busy streets tend to be related to one another in one of two ways.</p>
        `;
    },
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
        oldCity.style.backgroundImage = `url('assets/cities/cropped2/practice2.png')`;
        oldCity.style.backgroundSize = 'cover';
        oldCity.style.backgroundPosition = 'center';
        transitionContainer.appendChild(oldCity);

        // Create new city element
        let newCity = document.createElement('div');
        newCity.style.width = '50%'; // Half of the container
        newCity.style.height = '100%';
        newCity.style.backgroundImage = `url('assets/cities/cropped2/practice3.png')`;
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
    on_finish: function() {
    }
};

const instructions8 = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {
        const fontSize = "28px"; // Define font size as a variable
        return `
            <div class="instruction-section" style="z-index: 2000; position: relative;">
                <h1>How can you predict which intersections have tolls (or no tolls)?</h1>
                <p style="font-size: ${fontSize};">Each city has particular traffic properties, such that the busy streets tend to be related to one another in one of two ways.</p>
            </div>
            <div class="instruction-section"> 
                <h1>'Row cities'</h1>
                <p style="font-size: ${fontSize};">In row cities, the opposite is true: traffic tends to run from east to west every day, meaning that tolls tend to be clustered in rows.</p>
                <p style="font-size: ${fontSize};">That is, a row may have a lot of tolls, or not many tolls.</p>
                <p style="font-size: ${fontSize};">The particular locations of these busy rows may change each day, but the city will always have this row-dependent feature.</p>
                <p style="font-size: ${fontSize};">Press spacebar to continue.</p>
            </div>
            <div class="jobs-layout">
                <div class="instruction-section" style="text-align: center;  font-size: 20px; color: #3a3a3a;">
                    <h3><strong>Example day ${practice5TrialIndex + 1} tolls</strong><h3>
                </div>
                <div id="grid-container" class="current-job-section"></div>
            </div>
        `;
    },
    choices: [' '], // Wait for spacebar to continue
    on_load: function() {
        const gridContainer = document.getElementById('grid-container');
        const revealCosts = true; // Set to true to show costs
        gridContainer.innerHTML = practice5Grid.createBlankGridHTML(practice5TrialIndex, revealCosts); // Render a blank grid
        practice5TrialIndex++;
    },
    on_finish: function() {
    }
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
        window.scrollTo(0, 0); // Scrolls to the top of the page
        document.documentElement.requestFullscreen(); // Forces full-screen mode
    }
};


// instructions review - i.e. ask participants if they want to review the instructions pages (instructions1-instructions8, without the practices)
const instructionsReview = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {
        // document.body.style.zoom = "100%";
        document.body.style.zoom = zoomFactor;
        return `
            <div class="instruction-section">
                <h1>Review Instructions</h1>
                <p>We will now ask you a few questions to check your understanding of the task. Before doing so, you have the opportunity to review the instructions...</p>
            </div>

            <div class="instruction-section">
                <h1>Would you like to see the instructions again?</h1>
                <p>To review all the instructions from the beginning, press the backspace key.</p>
                <p>Otherwise, if you feel ready to continue, please press the spacebar.</p>
            </div>
        `;
    },
    choices: [' ', 'backspace'],
    on_load: function() {
    },
    on_finish: function(data) {
        // if (data.response === 'backspace') {
        //     // Restart the experiment by reloading the page
        //     location.reload();
        // }
    }
};

// Explanation of bonus
const instructions9 = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {
        const nGrids = grid.nGrids; // Retrieve the number of days from grid.nGrids
        return `
            <div class="instruction-section">
                <h1>Bonus Payment</h1>
                <p>Remember: your aim is to minimise the total cost paid each day by predicting which intersections will incur a toll, and hence by selecting jobs that you think will be least costly.</p>
                <p>This means that when choosing a job, it helps to think about which intersections you might visit later on in that day. These are highlighted in <span style="color: rgb(240, 110, 254);">pink</span>, and shown in your upcoming dispatches.</p>
                <p>At the end of the experiment, we will assess your performance by assessing how well you chose jobs that were the least costly on a randomly selected set of days and cities. This will determine whether you receive a bonus payment.</p>
                <p>So, you should pay attention throughout the experiment - i.e. on every day, and in every city.</p>
                <p>Remember also: you will have 8 seconds to select a job once the current dispatch turns yellow, otherwise the trial will timeout. If you timeout too many times, the experiment will end and you will return to Prolific.</p>
            </div>

            <div class="instruction-section">
                <h2>When you are ready to begin the experiment, press the spacebar.</h2>
            </div>
        `;
    },
    choices: [' '], // Spacebar to continue
    on_load: function() {
    },
    on_finish: function(data) {
    }
};

// Freetext feedback trial
const feedback_trial1 = {
    type: jsPsychSurveyText,
    preamble: `
        <div class="instruction-section">
            <h1>Q1: What strategy did you use to learn about the cities?</h1>
        </div>
    `,
    questions: [
        {
            prompt: `
            `,
            name: 'strategy',
            rows: 5,
            columns: 60,
            required: true
        },
    ],
    button_label: 'Submit'
};

const feedback_trial2 = {
    type: jsPsychSurveyText,
    preamble: `
        <div class="instruction-section">
            <h1>Q2: Did your strategy differ depending on whether you were in a row city or a column city?</h1>
        </div>
    `,
    questions: [
        {
            prompt: `
            `,
            name: 'observations',
            rows: 5,
            columns: 60,
            required: false
        },
    ],
    button_label: 'Submit'
};

const feedback_trial3 = {
    type: jsPsychSurveyText,
    preamble: `
        <div class="instruction-section">
            <h1>Q3: When choosing each job, how far ahead did you look at the upcoming jobs to make your decision?</h1>
        </div>
    `,
    questions: [
        {
            prompt: `
            `,
            name: 'lookahead',
            rows: 5,
            columns: 60,
            required: false
        }
    ],
    button_label: 'Submit'
};

const feedback_trial4 = {
    type: jsPsychSurveyText,
    preamble: `
        <div class="instruction-section">
            <h1>Q4: Any other comments or feedback about the experiment?</h1>
        </div>
    `,
    questions: [
        {
            prompt: `
            `,
            name: 'other_comments',
            rows: 5,
            columns: 60,
            required: false
        }
    ],
    button_label: 'Submit'
};

function create_need_for_cognition(){

    const space_bar_message = "<p>[Press the space bar to continue.]</p>";
  
    const NFC_instructs = `
        <div class="jspsych-survey-multi-choice-question">
        <h2 style="font-size: 30;"> <span style="color: #ece75d;">For each of the statements below, please indicate whether or not the statement is characteristic of you or of what you believe.</span></h2>
        <h2 style="font-size: 30;"> <span style="color: #ece75d;">Note: you may need to scroll to see all of the questions.</span></h2>
        </div>`;

  
    var options = ["extremely uncharacteristic of me", "somewhat uncharacteristic of me", "uncertain", "somewhat characteristic of me", "extremely characteristic of me"];
  
    //var standards = {options: options, required:false, horizontal:true};
    var question_list = ["1. I prefer complex to simple problems.",
      "2. I like to have the responsibility of handling a situation that requires a lot of thinking.",
      "3. Thinking is not my idea of fun.",
      "4. I would rather do something that requires little thought than something that is sure to challenge my thinking abilities.",
      "5. I try to anticipate and avoid situations where there is a likely chance I will have to think in depth about something.",
      "6. I find satisfaction in deliberating hard and for long hours.",
      "7. I only think as hard as I have to.",
      "8. I prefer to think about small daily projects to long term ones.",
      "9. I like tasks that require little thought once I've learned them.",
      "10. The idea of relying on thought to make my way to the top appeals to me.",
      "11. I really enjoy a task that involves coming up with new solutions to problems.",
      "12. Learning new ways to think doesn't excite me very much.",
      "13. I prefer my life to be filled with puzzles I must solve.",
      "14. The notion of thinking abstractly is appealing to me.",
      "15. I would prefer a task that is intellectual, difficult, and important to one that is somewhat important but does not require much thought.",
      "16. I feel relief rather than satisfaction after completing a task that requires a lot of mental effort.",
      "17. It's enough for me that something gets the job done; I don't care how or why it works.",
      "18. I usually end up deliberating about issues even when they do not affect me personally."]
  
      // make a list of dictionaries [{},{},{}] with standard settings
  
    const npages = 3;  
    const ends = [question_list.length/npages, 2*(question_list.length/npages), question_list.length];
    const formattedqs1 = [];
    for(var q=0; q<ends[0]; q++){
      formattedqs1.push({prompt: "<strong>" + question_list[q] + "</strong>", options: options, required:false, horizontal:true, name:"NFC"+question_list[q][0]+question_list[q][1]});
    }
    //add a Screener question
    formattedqs1.push({prompt: "<strong>7. Please select 'extremely characteristic of me' if you've read this question.</strong>", options: options, required:false, horizontal:true,name:'screener'})
    const formattedqs2 = [];
    for(var q=ends[0]; q<ends[1]; q++){
      formattedqs2.push({prompt: "<strong>" + question_list[q] + "</strong>", options: options, required:false, horizontal:true, name:"NFC"+question_list[q][0]+question_list[q][1]});
    }
    const formattedqs3 = [];
    for(var q=ends[1]; q<ends[2]; q++){
      formattedqs3.push({prompt: "<strong>" + question_list[q] + "</strong>", options: options, required:false, horizontal:true, name:"NFC"+question_list[q][0]+question_list[q][1]});
    }
      // standards['prompt'] = "This is a question?"
  
    // Define the NFC questionnaires with background color change
    var NFC1 = {
        timeline: [{
            type: jsPsychSurveyMultiChoice,
            questions: formattedqs1,
            preamble: NFC_instructs,
            data: { task: 'NFC' },
            on_load: function() {
                // Clear any existing background styles before applying a new one
                document.body.style.backgroundImage = '';
                document.body.style.backgroundSize = '';
                document.body.style.backgroundPosition = '';
                document.body.style.backgroundRepeat = '';
                document.body.style.backgroundColor = "black"; // Change background to black

                // Allow scrolling now
                document.body.style.overflowY = "auto";
                document.documentElement.style.overflowY = "auto";
            },
            on_finish: function(data) {
                // Object.assign(data, data.response);  // Adds each response directly to the trial data
                // delete data.response;
                document.body.style.backgroundColor = "";
            }            
        }]
    };

    var NFC2 = {
        timeline: [{
            type: jsPsychSurveyMultiChoice,
            questions: formattedqs2,
            preamble: NFC_instructs,
            data: { task: 'NFC' },
            on_load: function() {
                // Clear any existing background styles before applying a new one
                document.body.style.backgroundImage = '';
                document.body.style.backgroundSize = '';
                document.body.style.backgroundPosition = '';
                document.body.style.backgroundRepeat = '';
                document.body.style.backgroundColor = "black"; // Change background to black

                // Allow scrolling now
                document.body.style.overflowY = "auto";
                document.documentElement.style.overflowY = "auto";
            },
            on_finish: function(data) {
                // Object.assign(data, data.response);  // Adds each response directly to the trial data
                // delete data.response;
                document.body.style.backgroundColor = "";
            }            
        }]
    };

    var NFC3 = {
        timeline: [{
            type: jsPsychSurveyMultiChoice,
            questions: formattedqs3,
            preamble: NFC_instructs,
            data: { task: 'NFC' },
            on_load: function() {
                // Clear any existing background styles before applying a new one
                document.body.style.backgroundImage = '';
                document.body.style.backgroundSize = '';
                document.body.style.backgroundPosition = '';
                document.body.style.backgroundRepeat = '';
                document.body.style.backgroundColor = "black"; // Change background to black

                // Allow scrolling now
                document.body.style.overflowY = "auto";
                document.documentElement.style.overflowY = "auto";
            },
            on_finish: function(data) {
                // Object.assign(data, data.response);  // Adds each response directly to the trial data
                // delete data.response;
                document.body.style.backgroundColor = "";
                var ppt_data = jsPsych.data.get().json();
                send_incomplete(subject_id, ppt_data);
            }            
        }]
    };

    let questionnaireTimeline;
    return questionnaireTimeline = [NFC1, NFC2, NFC3];
  
  };

// Create timelines

function createEthicsTimeline() {
    const timeline = [];
    // Informed consent
    timeline.push(informedConsentTrial);
    timeline.push(dataProtectionTrial);
    timeline.push(fullscreenTrial);

    return timeline
}

function createInstructionsTimeline() {
    
    const timeline = [];
    
    // city assignments
    const numCities = data.env_costs.n_cities; // Assuming this is the number of cities
    createCityMapping(numCities);

    // Welcome message
    timeline.push(instructions1);

    // Practice selection
    timeline.push(instructions2);
    timeline.push(instructions2_5);
    // timeline.push(practice1SelectionTrial);
    // timeline.push(practice1AnimationTrial);
    // timeline.push(practice1SelectionTrial);
    // timeline.push(practice1AnimationTrial);

    // Practice a full day
    // timeline.push(instructions3);
    // timeline.push(instructions3_1);
    // timeline.push(instructions3_2);
    // timeline.push(instructions3_3);
    // timeline.push(instructions3_4);
    // timeline.push(instructions3_5);
    // timeline.push(instructions3_6);
    // timeline.push(instructions3_7);

    timeline.push(instructions3_node);
    timeline.push(practiceFirstDayTrial);
    for (let i = 0; i < grid.nTrials; i++) {
        // timeline.push(practice3PreSelectionTrial);
        timeline.push(practice3SelectionTrial);
        timeline.push(practice3AnimationTrial);
    }
    timeline.push(practiceGridFeedback);

    // // Animation to show grid resetting, and then another day
    timeline.push(instructions4);
    timeline.push(practiceFirstDayTrial);
    for (let i = 0; i < grid.nTrials; i++) {
        // timeline.push(practice3PreSelectionTrial);
        timeline.push(practice3SelectionTrial);
        timeline.push(practice3AnimationTrial);
    }
    timeline.push(practiceGridFeedback);

    // New city animation
    timeline.push(instructions5);

    // illustrate contexts
    for (let i = 1; i <= grid.nGrids; i++) {
        timeline.push(instructions6);
    }
    timeline.push(instructions7);
    for (let i = 1; i <= grid.nGrids; i++) {
        timeline.push(instructions8);
    }

    // Add the option to review the instructions
    timeline.push(instructionsReview);

    // timeline.push(feedback_trial1);
    // timeline.push(feedback_trial2);
    // timeline.push(feedback_trial3);
    // timeline.push(feedback_trial4);
    
    return timeline
}


// Understanding checks
function createQuizTimeline() {
    const timeline = [];
    const quizTrials = createQuizTrials(jsPsych);
    timeline.push(...quizTrials);
    timeline.push(instructions9)
    return timeline
}

// bonus message

function createMainTimeline() {
    const timeline = [];

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
        // timeline.push(pathPreSelectionTrial);
        timeline.push(pathSelectionTrial);
        timeline.push(pathAnimationTrial);
    }

    // Add the end and bonus message
    timeline.push(end);
    timeline.push(feedback_trial1);
    timeline.push(feedback_trial2);
    timeline.push(feedback_trial3);
    timeline.push(feedback_trial4);
    
    // questionnaire
    timeline.push(preQuestionnaire);
    var NFC_timeline = {
        timeline: create_need_for_cognition(),  
    } 
    timeline.push(NFC_timeline);
    timeline.push(bonus);

    return timeline;
}

// Start experiment when the page loads
function initializeExperiment() {
    
    // ethics timeline
    const ethicsTimeline = createEthicsTimeline();
  
    // instructions timeline
    const instructionTimeline = createInstructionsTimeline();
  
    // wrap instructions in a loop node
    const instructionsLoop = {
      timeline: instructionTimeline,
      loop_function: function() {
        const lastChoice = jsPsych.data.get().last(1).values()[0].response;
        if (lastChoice === 'backspace') {
            console.log('repeating instructions');
            return true;
        } else {
            return false;
        }
      }
    };
  
    // quiz timeline
    const quizTimeline = createQuizTimeline();

    // main experiment timeline
    const mainTimeline = createMainTimeline();
  
    // Combine everything into a single timeline
    const fullTimeline = [
    //   ...ethicsTimeline,
      instructionsLoop,
      ...quizTimeline,
      ...mainTimeline
    ];
  
    // Run it all at once
    jsPsych.run(fullTimeline);
  }
  