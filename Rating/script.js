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

    dialogueData.conversations.forEach((conversation, convIndex) => {
        const dialogueDiv = document.createElement('div');
        dialogueDiv.className = 'dialogue';

        // 显示图像（如果有）
        if (conversation.images && conversation.images.length > 0) {
            conversation.images.forEach(image => {
                const img = document.createElement('img');
                img.src = image_root+image;
                dialogueDiv.appendChild(img);
            });
        }

        // 显示用户输入
        const userInput = document.createElement('p');
        userInput.textContent = `用户输入: ${conversation.user_input}`;
        dialogueDiv.appendChild(userInput);

        // 显示模型预测
        const prediction = document.createElement('div');
        prediction.className = 'prediction';
        prediction.textContent = `模型预测: ${conversation.prediction}`;
        dialogueDiv.appendChild(prediction);

        // 显示标签
        const label = document.createElement('div');
        label.className = 'label';
        label.textContent = `标签: ${conversation.label}`;
        dialogueDiv.appendChild(label);

        // 评分选项
        const ratingDiv = document.createElement('div');
        ratingDiv.className = 'rating';
        for (let i = 1; i <= 5; i++) {
            const label = document.createElement('label');
            label.innerHTML = `<input type="radio" name="rating-${convIndex}" value="${i}" 
                ${ratings[currentDialogueIndex][convIndex] === i ? 'checked' : ''}> ${i}分`;
            ratingDiv.appendChild(label);
        }
        dialogueDiv.appendChild(ratingDiv);

        dialoguesContainer.appendChild(dialogueDiv);
    });
}

// 保存当前对话的评分
function saveRatings() {
    const currentRatings = {};
    dialogueList[currentDialogueIndex].conversations.forEach((conversation, convIndex) => {
        const selectedRating = document.querySelector(`input[name="rating-${convIndex}"]:checked`);
        if (selectedRating) {
            currentRatings[convIndex] = parseInt(selectedRating.value);
        }
    });
    ratings[currentDialogueIndex] = currentRatings; // 更新评分数据
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
        ratings: ratings[index]
    }));

    // 生成 JSON 文件
    const dataStr = JSON.stringify(result, null, 2);
    const blob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);

    // 创建下载链接
    const a = document.createElement('a');
    a.href = url;
    a.download = 'ratings.json';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    alert("评分已保存并下载！");
}