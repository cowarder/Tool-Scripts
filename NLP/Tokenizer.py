from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

#Tokenizer将每一个word用一个数字表示
max_features = 50000
tk = Tokenizer(lower = True, num_words = max_features)
full_text = list(train.question_text.values) + list(test.question_text.values)
tk.fit_on_texts(full_text)

train_tokenized = tk.texts_to_sequences(train.question_text.fillna('missed'))
test_tokenized = tk.texts_to_sequences(test.question_text.fillna('missed'))

#将Tokenize之后的数据转换为相同长度
max_len = 70
X_train = pad_sequences(train_tokenized, maxlen=max_len)
X_test = pad_sequences(test_tokenized, maxlen=max_len)

	