index.html: 웹 페이지 구동 파일(front)
app.py: 모델 구현 파일(back)

실행 방법(vscode에서)
1. 필요한 라이브러리 설치: 아래 명령을 터미널에 입력
	- 최소: flask, flask-cors, numpy
	- CSV 기반 결과 사용 시: pandas (선택)

```
pip install flask flask-cors numpy pandas
```
2. app.py 실행: 터미널에 python app.py 입력
3. 브라우저에서 열기: Live Server 대신 http://127.0.0.1:5000 로 접속 (같은 오리진에서 /api/config, /simulate 모두 동작)
