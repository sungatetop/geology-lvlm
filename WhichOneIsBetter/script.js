// script.js

let dialogueList = []; // 存储加载的对话数据列表
let currentDialogueIndex = 0; // 当前显示的对话索引
let ratings = []; // 存储评分数据

// 加载本地 JSONL 文件
function loadData() {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    if (!file) {
        alert("请选择一个 JSONL 文件！");
        return;
    }

    const reader = new FileReader();
    reader.onload = function (e) {
        const text = e.target.result;
        const lines = text.split('\n').filter(line => line.trim() !== ''); // 按行分割并过滤空行
        dialogueList = lines.map(line => JSON.parse(line)); // 解析每一行 JSON
        currentDialogueIndex = 0; // 重置当前索引
        ratings = dialogueList.map(() => ({})); // 初始化评分数据
        showDialogue(currentDialogueIndex); // 显示第一个对话
    };
    reader.readAsText(file);
}

// 显示指定索引的对话数据
function showDialogue(index) {
    if (index < 0 || index >= dialogueList.length) {
        alert("没有更多对话了！");
        return;
    }
    const image_root="../eval_origin/"
    currentDialogueIndex = index;
    const dialogueData = dialogueList[index];
    const dialoguesContainer = document.getElementById('dialogues');
    dialoguesContainer.innerHTML = ''; // 清空之前的内容

    // 显示用户输入
    const userInput = document.createElement('p');
    userInput.textContent = `用户输入: ${dialogueData.conversations[0].user_input}`;
    dialoguesContainer.appendChild(userInput);

    // 显示图片（如果有）
    if (dialogueData.conversations[0].images && dialogueData.conversations[0].images.length > 0) {
        dialogueData.conversations[0].images.forEach(image => {
            const img = document.createElement('img');
            img.src =image_root+image;
            dialoguesContainer.appendChild(img);
        });
    }

    // 随机打乱模型预测和标签的顺序
    const options = [
        { text: dialogueData.conversations[0].prediction, type: 'prediction' },
        { text: dialogueData.conversations[0].label, type: 'label' }
    ];
    shuffleArray(options); // 随机打乱顺序

    // 显示两个选项
    options.forEach((option, optionIndex) => {
        const optionDiv = document.createElement('div');
        optionDiv.className = 'dialogue';

        // 显示选项内容
        const optionText = document.createElement('div');
        optionText.className = option.type === 'prediction' ? 'prediction' : 'label';
        optionText.textContent = `选项 ${optionIndex + 1}: ${option.text}`;
        optionDiv.appendChild(optionText);

        // 评分选项
        const ratingDiv = document.createElement('div');
        ratingDiv.className = 'rating';
        ratingDiv.innerHTML = `
            <label>
                <input type="radio" name="rating-${currentDialogueIndex}" value="${optionIndex + 1}"> 更好
            </label>
            <label>
                <input type="radio" name="rating-${currentDialogueIndex}" value="neither"> 都不好
            </label>
        `;
        optionDiv.appendChild(ratingDiv);

        dialoguesContainer.appendChild(optionDiv);
    });
}

// 保存当前对话的评分
function saveRatings() {
    const selectedRating = document.querySelector(`input[name="rating-${currentDialogueIndex}"]:checked`);
    if (selectedRating) {
        ratings[currentDialogueIndex] = selectedRating.value;
    }
}

// 切换到上一个对话
function prevDialogue() {
    saveRatings(); // 保存当前对话的评分
    if (currentDialogueIndex > 0) {
        currentDialogueIndex--;
        showDialogue(currentDialogueIndex);
    } else {
        alert("已经是第一个对话了！");
    }
}

// 切换到下一个对话
function nextDialogue() {
    saveRatings(); // 保存当前对话的评分
    if (currentDialogueIndex < dialogueList.length - 1) {
        currentDialogueIndex++;
        showDialogue(currentDialogueIndex);
    } else {
        alert("已经是最后一个对话了！");
    }
}

// 提交评分并生成下载文件
function submitRatings() {
    saveRatings(); // 保存当前对话的评分

    // 构建评分结果
    const result = dialogueList.map((dialogue, index) => ({
        id: dialogue.id,
        rating: ratings[index] || '未评分'
    }));

    // 生成 JSON 文件
    const dataStr = JSON.stringify(result, null, 2);
    const blob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);

    // 创建下载链接
    const a = document.createElement('a');
    a.href = url;
    a.download = 'blind_ratings.json';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    alert("评分已保存并下载！");
}

// 随机打乱数组
function shuffleArray(array) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
}