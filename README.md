The AI Cheat Detector System is an intelligent application designed to detect cheating in uploaded interview videos. It uses a combination of video, audio, and AI-based analysis to identify suspicious activities and provide a clear final decision.

The system includes several important features such as face detection to identify absence or multiple faces, phone detection using YOLOv8, and audio analysis to detect suspicious voice activity.An alert-based scoring system is used to calculate the cheating probability, and a dashboard is provided to store and view past results.

The project is built using multiple technologies. OpenCV is used for video processing, MediaPipe is used for face detection, and YOLOv8 is used for phone detection. Streamlit is used to create the user interface, while MySQL is used for storing user data and cheating logs. MoviePy is used for extracting and analyzing audio from the video.

The project structure is simple and easy to understand. The main file is app.py, which contains the complete logic of the system. A MySQL database is used to store user login details and analysis results. The YOLOv8 model required for object detection is automatically downloaded during execution, so no manual setup is required for it.

To run the project, the required Python libraries need to be installed, and a MySQL database named ai_cheat_detector should be created. The database credentials can be updated in the code if needed.A Gemini API key can be set as an environment variable for AI-based decision making. The application can then be started using the Streamlit command, and accessed through a web browser using the default login credentials.

The system works in a step-by-step flow. First, the user uploads a video. The system then performs face detection, phone detection, and audio analysis. Based on these checks, alerts are generated and a cheating score is calculated. Finally, an AI-based decision is produced, and the results are stored in the database for future reference.

The output of the system includes the cheating score, probability, and risk level categorized as low, medium, or high. It also provides a final decision such as clean, suspicious, or cheating, along with clear reasons and alerts, making the result easy to understand.

This project can be further improved by adding real-time monitoring, enhancing audio analysis, and improving behavior detection accuracy. The system is developed for educational purposes to demonstrate how AI can be used for cheating detection.

Team Members: Vishnupriya K.M, Hamsa S.M, Raheela Banu M.R, Shaik Shekha, Nandini R.R, Shreeraksha Girish Kulkarni
