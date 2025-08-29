from aiogram import Bot, Dispatcher, F
from aiogram.filters.command import Command
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery, Message, FSInputFile
import cv2, numpy as np
import os, sys
from cv2 import Mat
import asyncio
import warnings
import pickle
from catboost import CatBoostClassifier
import torch
import torchvision
import torch.nn as nn

warnings.filterwarnings("ignore", category=UserWarning)


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(3, 16, kernel_size=7, stride=1, padding=3),
                                    nn.ReLU(), nn.BatchNorm2d(16), nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
                                    nn.ReLU(), nn.BatchNorm2d(32), nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(8 * 8 * 64, 1000)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1000, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.relu(self.fc1(out))
        out = self.drop_out(self.relu(self.fc2(out)))
        out = self.fc3(out)
        return out

with open("randomForest.pkl", "rb") as f:
    RF = pickle.load(f)
catboost = CatBoostClassifier()
catboost.load_model('catboost.cbm')
model = ConvNet()
model.load_state_dict(torch.load('conv_net_model_new.ckpt', weights_only=True))


def coga(_image):
    global model
    model.eval()
    with torch.no_grad():
        out = model(_image)
        predicted = (nn.Softmax()(out.data)[:, 1]).to(float)
    return predicted

TOKEN = 

lungs_dict_v2 = {'AGE': 'Какой у вас возраст?', 'GENDER': 'Какой у вас пол?', 'SMOKING': 'Вы курите?',
     'YELLOW_FINGERS': 'Есть ли у вас желтизна на пальцах?', 'ANXIETY': 'Часто ли вы испытываете чувство тревоги?',
     'PEER_PRESSURE': 'Ваше артериальное давление в норме?', 'CHRONIC_DISEASE': 'Имеются ли у вас хронические заболевания?',
     'FATIGUE': 'Чувствуете ли вы хроническую усталость?', 'ALLERGY': 'Есть аллергия?',
     'WHEEZING': 'Замечали ли в последнее время хрипоту?', 'ALCOHOL_CONSUMING': 'Употребляете алкоголь?',
     'COUGHING': 'Тревожит ли вас кашель?', 'SHORTNESS_OF_BREATH': 'Страдаете ли вы одышкой?',
     'SWALLOWING_DIFFICULTY': 'Испытываете трудность при глотании?', 'CHEST_PAIN': 'Присутствует боль в груди?'}

lungs_dict_var = {
    'AGE': [[("20", "< 20"), ("30", "21-40"), ("50", "41-60"), ("70", "> 60")]],
    'GENDER': [[("0", "Мужчина"), ("1", "Женщина")]],
    'SMOKING': [[("1", "Не курю"), ("2", "Курю")]],
    'YELLOW_FINGERS': [[("1", "Нет"), ("2", "Есть")]],
    'ANXIETY': [[("2", "Часто"), ("1", "Редко / не чувствую")]],
    'PEER_PRESSURE': [[("1", "В норме"), ("2", "Не в норме")]],
    'CHRONIC_DISEASE': [[("1", "Нет"), ("2", "Есть")]],
    'FATIGUE': [[("1", "Не чувствую"), ("2", "Есть такое")]],
    'ALLERGY': [[("1", "Нет"), ("2", "Есть")]],
    'WHEEZING': [[("1", "Нет"), ("2", "Похрипываю")]],
    'ALCOHOL_CONSUMING': [[("1", "Не пью"), ("2", "Побухиваю иногда")]],
    'COUGHING': [[("1", "Не кашляю"), ("2", "Кашляю")]],
    'SHORTNESS_OF_BREATH': [[("1", "Нет"), ("2", "Да")]],
    'SWALLOWING_DIFFICULTY': [[("1", "Нет"), ("2", "Да")]],
    'CHEST_PAIN': [[("1", "Нет"), ("2", "Да")]]
}

lungs_dict_index = {
    'AGE': 'GENDER ',
    'GENDER': 'SMOKING ',
    'SMOKING': 'YELLOW_FINGERS ',
    'YELLOW_FINGERS': 'ANXIETY ',
    'ANXIETY': 'PEER_PRESSURE ',
    'PEER_PRESSURE': 'CHRONIC_DISEASE ',
    'CHRONIC_DISEASE': 'FATIGUE ',
    'FATIGUE': 'ALLERGY ',
    'ALLERGY': 'WHEEZING ',
    'WHEEZING': 'ALCOHOL_CONSUMING ',
    'ALCOHOL_CONSUMING': 'COUGHING ',
    'COUGHING': 'SHORTNESS_OF_BREATH ',
    'SHORTNESS_OF_BREATH': 'SWALLOWING_DIFFICULTY ',
    'SWALLOWING_DIFFICULTY': 'CHEST_PAIN ',
    'CHEST_PAIN': 'None '}

lungs_dict_index_ret = {
    'AGE': "None",
    'GENDER': 'AGE',
    'SMOKING': 'GENDER',
    'YELLOW_FINGERS': 'SMOKING',
    'ANXIETY': 'YELLOW_FINGERS',
    'PEER_PRESSURE': 'ANXIETY',
    'CHRONIC_DISEASE': 'PEER_PRESSURE',
    'FATIGUE': 'CHRONIC_DISEASE',
    'ALLERGY': 'FATIGUE',
    'WHEEZING': 'ALLERGY',
    'ALCOHOL_CONSUMING': 'WHEEZING',
    'COUGHING': 'ALCOHOL_CONSUMING',
    'SHORTNESS_OF_BREATH': 'COUGHING',
    'SWALLOWING_DIFFICULTY': 'SHORTNESS_OF_BREATH',
    'CHEST_PAIN': 'SWALLOWING_DIFFICULTY'}

user_img_size = 150

test_v2_index = {
    "Age": "Gender ",
    "Gender": "BMI ",
    "BMI": "Smoking ",
    "Smoking": "PhysicalActivity ",
    "PhysicalActivity": "AlcoholIntake ",
    "AlcoholIntake": "None "
}

test_v2_index_ret = {
    "Gender": "Age",
    "BMI": "Gender",
    "Smoking": "BMI",
    "PhysicalActivity": "Smoking",
    "AlcoholIntake": "PhysicalActivity"
}

test_v2 = {
    "Age": "1. Выберите вашу возрастную категорию",
    "Gender": "2. Вы мужчина или женщина?",
    "BMI": "3. Выберите категорию вашего ИМТ\n"
           "ИМТ = вес / (рост * рост)\n"
           "Рост в метрах, вес - в килограммах",
    "Smoking": "Курите или нет?",
    "PhysicalActivity": "Сколько часов в неделю уделяете спорту?",
    "AlcoholIntake": "Сколько юнитов алкоголя выпиваете за неделю?\n"
                     "Количество юнитов вычисляется по формуле:\n"
                     "Объём напитка * процент алкоголя в нём"
}

test_v2_var = {
    "Age": [[("20", "< 20"), ("30", "21-40"), ("50", "41-60"), ("70", "> 60")]],
    "Gender": [[("0", "Мужчина"), ("1", "Женщина")]],
    "BMI": [
        [("15", "< 20"), ("20", "20-25"), ("25", "25-30")],
        [("30", "30-35"), ("35", "35-40"), ("40", "> 40")]
    ],
    "Smoking": [[("0", "Не курю"), ("1", "Курю")]],
    "PhysicalActivity": [
        [("0", "< часа"), ("1", "1-2 часа"), ("2", "2-3 часа")],
        [("3", "3-4 часа"), ("4", "4-5 часов"), ("5", "5-6 часов")],
        [("6", "6-7 часов"), ("7", "7-8 часов"), ("8", "8-9 часов")],
        [("9", "9-10 часов"), ("10", "> 10 часов")]
    ],
    "AlcoholIntake": [
        [("0", "Не пью вообще"), ("1", "1 юнит")],
        [("2", "2 юнита"), ("3", "3 юнита"), ("4", "4 юнита")],
        [("5", "5 юнитов и более")]
    ]
}

gen = "check_cancer_v2 "
for i in test_v2.keys():
    for x in range(len(test_v2_var[i])):
        for y in range(len(test_v2_var[i][x])):
            _r, _q = test_v2_var[i][x][y]
            _r = gen + test_v2_index[i] + _r
            test_v2_var[i][x][y] = (_q, _r)

gen_l = "check_lungs_v2 "
for i in lungs_dict_v2.keys():
    for x in range(len(lungs_dict_var[i])):
        for y in range(len(lungs_dict_var[i][x])):
            _r, _q = lungs_dict_var[i][x][y]
            _r = gen_l + lungs_dict_index[i] + _r
            lungs_dict_var[i][x][y] = (_q, _r)

bot = Bot(TOKEN)
dp = Dispatcher()

params = cv2.SimpleBlobDetector.Params()
params.filterByArea = True
params.minArea = 500
params.filterByCircularity = True
params.minCircularity = 0.2
params.filterByConvexity = True
params.minConvexity=0.7
params.filterByInertia = False
detector_large = cv2.SimpleBlobDetector.create(params)
params.minArea = 230
params.maxArea = 500
params.minCircularity = 0.6
params.minConvexity = 0.8
detector_small = cv2.SimpleBlobDetector.create(params)

lungs_dict = dict()
cancer_dict = dict()
skin_dict = dict()


inline_builder_column = lambda buttons: InlineKeyboardMarkup(inline_keyboard=[[
    InlineKeyboardButton(text = t, callback_data= d)] for t, d in buttons])

inline_builder_row = lambda buttons: InlineKeyboardMarkup(inline_keyboard=[[
    InlineKeyboardButton(text = t, callback_data= d) for t, d in buttons]])

inline_builder_matrix = lambda buttons: InlineKeyboardMarkup(inline_keyboard=[[
    InlineKeyboardButton(text = t, callback_data= d) for t, d in b] for b in buttons])

async def imageCrop(img: Mat):
    img_ = cv2.GaussianBlur(img, (3, 3), 0)
    hsv = cv2.cvtColor(img_, cv2.COLOR_RGB2HSV)  # Convert to hsv color system
    h, s, v = cv2.split(hsv)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3, 3))
    v = clahe.apply(v)
    img_ = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2RGB)

    gray = cv2.cvtColor(img_, cv2.COLOR_RGB2GRAY)
    mean = int(np.mean(gray)+np.average(gray))//2
    T = cv2.threshold(gray, round(mean*0.8), 255, cv2.THRESH_BINARY)[1]

    kpoints = detector_large.detect(T)
    pts = list(kpoints)
    _kpoints = detector_small.detect(T)
    pts += list(_kpoints)

    return pts, kpoints, _kpoints


@dp.message(Command("start"))
async def start(message: Message):
    await message.answer(f"Привет, {message.from_user.username}! Я помогу тебе определить, с какой вероятностью у тебя онкология",
                         reply_markup=inline_builder_column([("Тест на общий риск рака", f"{gen}Age None"),
                                                             ("Тест на рак легких", f"{gen_l}AGE None"),
                                                             ("Определение рака кожи", "check_skin")]))


@dp.callback_query(F.data.startswith(gen))
async def lungs(call: CallbackQuery):
    global cancer_dict, d
    gen, key, ret = call.data.split(" ")
    if key == "None":
        await call.message.answer("Тест пройден!")
        cancer_dict[call.message.chat.id]["AlcoholIntake"] = int(ret)
        pred = RF.predict_proba([list(cancer_dict[call.message.chat.id].values())])[0][1]*100
        cancer_dict.pop(call.message.chat.id, None)
        await call.message.answer(f"Ваш риск развития рака (в будущем): {pred:.2f}%\n\nВНИМАНИЕ!!! Наш бот НЕ является врачом, "
                                  f"диагнозы НЕ ставит и предоставляет лишь примерную информацию! За точным "
                                  f"обследованием обращайтесь к специалистам")
        await call.message.delete()
    elif key == "Age":
        cancer_dict[call.message.chat.id] = dict()
        await call.message.answer(test_v2[key], reply_markup=inline_builder_matrix(test_v2_var[key]))
        await call.message.edit_reply_markup(reply_markup=None)
    else:
        await call.message.edit_text(test_v2[key], reply_markup=inline_builder_matrix(test_v2_var[key]))
        cancer_dict[call.message.chat.id][test_v2_index_ret[key]] = int(ret)


@dp.callback_query(F.data.startswith(gen_l))
async def lungs(call: CallbackQuery):
    global lungs_dict, d
    gen, key, ret = call.data.split(" ")
    if key == "None":
        await call.message.answer("Тест пройден!")
        await call.message.delete()
        lungs_dict[call.message.chat.id]["CHEST_PAIN"] = int(ret)
        pred = catboost.predict_proba([list(lungs_dict[call.message.chat.id].values())])[0][1] * 100
        pred -= 0.3 if pred >= 0.5 else 0
        print(pred)
        print(type(pred))
        lungs_dict.pop(call.message.chat.id, None)
        await call.message.answer(
            f"Ваш риск развития рака лёгких: {pred:.2f}%\n\nВНИМАНИЕ!!! Наш бот НЕ является врачом, "
            f"диагнозы НЕ ставит и предоставляет лишь примерную информацию! За точным "
            f"обследованием обращайтесь к специалистам")
        await call.message.delete()
    elif key == "AGE":
        lungs_dict[call.message.chat.id] = dict()
        await call.message.answer(lungs_dict_v2[key], reply_markup=inline_builder_matrix(lungs_dict_var[key]))
        await call.message.edit_reply_markup(reply_markup=None)
    else:
        await call.message.edit_text(lungs_dict_v2[key], reply_markup=inline_builder_matrix(lungs_dict_var[key]))
        lungs_dict[call.message.chat.id][lungs_dict_index_ret[key]] = int(ret) if ret.isdigit() else ret


@dp.callback_query(F.data == "check_skin")
async def skin(call: CallbackQuery):
    await call.message.answer("Отправь фото исследуемой родинки\n\nВНИМАНИЕ\nЕсли обрезанная область не соответствует "
                              "исследуемой родинке, нажмите \"нет\" под ней. Если при этом Вас не удовлетворяет ни одна "
                              "найденная родинка - попробуйте отправить другое фото (под другим углом, с другим освещением)")
    await call.message.edit_reply_markup(reply_markup=None)


@dp.callback_query(F.data.startswith("SKIN"))
async def skin_(call: CallbackQuery):
    _, filename, status = call.data.split()
    await call.message.edit_reply_markup(reply_markup=None)
    if status == "1":
        img = np.array(cv2.imread(filename))
        os.remove(filename)
        if img.shape != (64, 64, 3):
            img = cv2.resize(img, (64, 64))
        b, g, r = cv2.split(img)
        arr = torch.tensor(np.array([[r, g, b]]).astype(np.float32))
        result = float(coga(arr)[0])*100
        await call.message.answer(
            f"Ваш риск развития рака кожи для данной родинки: {result:.2f}%\n\nВНИМАНИЕ!!! Наш бот НЕ является врачом, "
            f"диагнозы НЕ ставит и предоставляет лишь примерную информацию! За точным "
            f"обследованием обращайтесь к специалистам")
    else:
        await call.message.delete()
        os.remove(filename)


@dp.message(F.photo)
async def process_photo(message: Message):
    skin_dict[message.chat.id] = dict()
    file_name = f"{message.photo[-1].file_id}.jpg"
    await bot.download(message.photo[-1].file_id, destination=file_name)
    img = cv2.imread(file_name)
    os.remove(f"{message.photo[-1].file_id}.jpg")
    ret, k, _k = await imageCrop(img)
    _img = cv2.drawKeypoints(img, k, np.zeros((1, 1)), (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    _img = cv2.drawKeypoints(_img, _k, np.zeros((1, 1)), (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    w, h, _ = img.shape
    if h > w:
        h, w = w, h
    for kp in ret:
        x, y = kp.pt
        _xi = round(max(x-user_img_size, 0))
        _xm = round(min(x+user_img_size, w))
        _yi = round(max(y-user_img_size, 0))
        _ym = round(min(y + user_img_size, h))
        __img = _img[_yi:_ym, _xi:_xm].copy()
        skin_dict[message.chat.id][ret.index(kp)] = (kp, f"{message.chat.id} {ret.index(kp)}")
        cv2.imwrite(f"{message.chat.id}_{ret.index(kp)}.png", __img)
    for kp in ret:
        photo = FSInputFile(f"{message.chat.id}_{ret.index(kp)}.png", "rb")
        s = kp.size if kp.size > 32 else 32
        x, y = kp.pt
        await message.answer_photo(photo, caption="Начать распознавание этой родинки?",
                                   reply_markup=inline_builder_row([
                                       ("Да", f"SKIN {message.chat.id}_{ret.index(kp)}.png 1"),
                                       ("Нет", f"SKIN {message.chat.id}_{ret.index(kp)}.png 0")
                                   ]))
        _img = img[round(max(y - s, 0)):round(min(y + s, h)), round(max(x - s, 0)):round(min(x + s, w))]
        cv2.imwrite(f"{message.chat.id}_{ret.index(kp)}.png", _img)
    if len(ret) == 0:
        await message.answer("Видимо, родинку я не нашёл. Попробуйте отправить другое фото (под другим углом, с "
                                  "другим освещением)")


@dp.message()
async def handler(message: Message):
    await message.answer("Прости, я тебя не понял")


async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
