Tôi muốn viết 1 đoạn script python (có nhận đầu vào tham số CLI), có thể yêu cầu thêm tải thư viện thứ 3 nếu cầu (ví dụ huggingface_hub) nhưng ưu tiên built-in library của python để làm thực hiện 4 phase: Phase pull LLM (bao gồm download, tách phần), phase push docker (Viết Dockerfile cho từng phần, đánh tag tương ứng và push các phần), phase pull dockerhub các phần và phase restore LLM tái tạo lại repo ban đầu. Script nhận đầu vào là tên phase chạy riêng lẻ và các tham số cần thiết cho từng phase, ví dụ phase pull LLM thì cần tham số là link huggingface (yêu cầu), phase push docker thì cần tham số là max_docker_image_size, v.v. Script có thể nhận đầu vào là multiple phase để chạy liên tiếp, cách nhau bởi dấu phẩy, và các tham số yêu cầu của từng phase (hiện tại chỉ có phase đầu tiên là cần tham số đầu vào là URL) ví dụ: `python script.py --phases pull_llm,push_docker --huggingface_link https://huggingface.co/Qwen/Qwen3.6-27B/tree/main`

Mô tả các phase như sau:

# 1. Phase pull LLM

1. Tôi muốn tải được file và folder LLM từ input là link dạng ví dụ: https://huggingface.co/Qwen/Qwen3.6-27B/tree/main của huggingface về máy của tôi. Lưu vào folder `model_weights/<model_name>/raw/....`. Ignore folder `model_weights/<model_name>/raw/....` này
2. Sau khi tải về rồi, tôi muốn lưu vào 1 file `state.json` trong folder `model_weights/<model_name>/`, không ignore file này. File `state.json` sẽ chứa thông tin về các file đã tải về, bao gồm tên file, kích thước, và đường dẫn lưu trữ. Ví dụ:

```json
{
  "model_name": "<model_name>",
  "url": "https://huggingface.co/Qwen/Qwen3.6-27B/tree/main",
  "raw": [
    {
      "filename": "model.bin",
      "size": "12GB",
      "path": "model_weights/<model_name>/raw/model.bin"
    },
    {
      "filename": "config.json",
      "size": "1KB",
      "path": "model_weights/<model_name>/raw/config.json"
    }
  ]
}
```

3. Sau khi tải về rồi, tôi muốn tách các file có trong repo Hugging Face này, file lớn nhất cho phép là 5GB (<max_part_size> mặc định 5GB, có thể định nghĩa bằng biến trong code, biến môi trường, hoặc tham số CLI). Các phần này sẽ được lưu vào folder `model_weights/<model_name>/parts/....`. Ví dụ, nếu có một file có kích thước 12GB, nó sẽ được tách thành 3 phần: part1 (5GB), part2 (5GB), part3 (2GB). Cách đặt tên file các part sẽ là `<raw_filename>-part1.<extension>`, `<raw_filename>-part2.<extension>`, v.v. Ignore folder `model_weights/<model_name>/parts/....` này.
4. Sau khi tách xong file part này thì tiếp tục cập nhật vào file `state.json` để lưu thông tin về các phần đã tách ra, bao gồm tên file part, kích thước, và đường dẫn lưu trữ. Ví dụ:

```json
{
  "model_name": "<model_name>",
  "url": "https://huggingface.co/Qwen/Qwen3.6-27B/tree/main",
  "max_part_size": "<max_part_size>",
  "raw": [
    {
      "filename": "model.bin",
      "size": "12GB",
      "path": "model_weights/<model_name>/raw/model.bin"
    },
    {
      "filename": "config.json",
      "size": "1KB",
      "path": "model_weights/<model_name>/raw/config.json"
    }
  ],
  "parts": [
    {
      "raw_filename": "model.bin",
      "part_filename": "model.bin-part1.bin",
      "size": "5GB",
      "path": "model_weights/<model_name>/parts/model.bin-part1.bin"
    },
    {
      "raw_filename": "model.bin",
      "part_filename": "model.bin-part2.bin",
      "size": "5GB",
      "path": "model_weights/<model_name>/parts/model.bin-part2.bin"
    },
    {
      "raw_filename": "model.bin",
      "part_filename": "model.bin-part3.bin",
      "size": "2GB",
      "path": "model_weights/<model_name>/parts/model.bin-part3.bin"
    }
  ]
}
```

# 2. Phase push docker

1. Tôi muốn viết dockerfile, mỗi dockerfile có thể bao gồm 1 hoặc nhiều file parts đã tách từ trước, đảm bảo sao cho mỗi dockerfile không vượt quá <max_docker_image_size> (mặc định 5GB, có thể định nghĩa bằng biến trong code, biến môi trường, hoặc tham số CLI). Ví dụ, nếu `max_docker_image_size` là 10GB, thì tôi có thể gộp part1 và part2 vào cùng một dockerfile, còn part3 sẽ được đặt trong một dockerfile riêng. Cách đặt tên cho các dockerfile sẽ là `Dockerfile.part1`, `Dockerfile.part2`, v.v. lưu trong folder `model_weights/<model_name>/dockerfiles/....`. Mỗi dockerfile sẽ được sinh từ template trong code, chỉ khác nhau các câu lệnh copy để copy các file part tương ứng vào image, Sử dụng base image là `alpine:3.22.4`, không cài thêm hoặc update gì cả, chỉ đơn giản là copy file part vào image.
2. Sau khi viết xong dockerfile thì cập nhật danh sách dockerfile và các part mà nó bao gồm vào `state.json`
3. Sau đó sinh các tag tương ứng với mỗi dockerfile, ví dụ full image name: `hieucien/<model_name>:part1`, `hieucien/<model_name>:part2`, v.v. Cập nhật danh sách tag này vào `state.json`
4. Thực hiện build image song song (theo số core của máy) và cập nhật trạng thái image nào đã được build xong vào `state.json`
5. Sau khi build xong tất cả thì push tất cả các image này lên Docker Hub và cập nhật trạng thái image nào đã được push xong vào `state.json`

# 3. Phase pull dockerhub

1. Từ danh sách tag đã lưu trong `state.json`, và trạng thái đã build, đã push, thực hiện pull tất cả các image này từ Docker Hub về máy của tôi, cập nhật trạng thái image nào đã được pull xong vào `state.json`

# 4. Phase restore LLM

1. Sau khi pull xong tất cả các image này về máy, thực hiện extract các file part từ các image này ra folder `model_weights/<model_name>/extracted/parts/....`. Giữ nguyên tên và nội dung của các file được extract. Đảm bảo các file được extract phải đầy đủ theo đúng danh sách đã tách part ở phase 1, v.v. Ignore folder `model_weights/<model_name>/extracted/parts....` này.
2. Sau khi extract xong các file part này rồi, cập nhật trạng thái đã extract xong vào `state.json`
3. Sau khi extract xong các file part này rồi, thực hiện gộp các file part này lại thành các file raw ban đầu và lưu vào folder `model_weights/<model_name>/extracted/restored/....`. Đảm bảo các file được gộp phải đầy đủ theo đúng danh sách đã tách part ở phase 1, v.v. Ignore folder `model_weights/<model_name>/extracted/restored/....` này.
