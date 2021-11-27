# 1d_barcode_detection
バーコード検出を実装したリポジトリ。  
[この論文](https://www.researchgate.net/profile/Abderrahmane-Namane/publication/318792856_Fast_Real_Time_1D_Barcode_Detection_From_Webcam_Images_Using_the_Bars_Detection_Method/links/597f4e13aca272d5681884a8/Fast-Real-Time-1D-Barcode-Detection-From-Webcam-Images-Using-the-Bars-Detection-Method.pdf)をベースに実装。

バーコードのデコードはOpenCVに実装されている[デコード機能](https://docs.opencv.org/4.5.4/dc/df7/classcv_1_1barcode_1_1BarcodeDetector.html)を利用している。

論文との主なロジックの差分は以下

- 前処理を「ガウシアンフィルタ + 二値化 + 輪郭抽出」ではなく、「DoGフィルタ + 二値化 + 輪郭抽出」に変更
- バーコードでない部分の誤検知を減らすため、バーコードのある1つのバーの重心から他のバーの重心までのベクトルと、バーの向きがほぼ直角になるかのバリデーションを導入
- バーコードでない部分の誤検知を減らすため、隣り合うバーとの距離が一定以内でないとバーの一部とみなさないようなバリデーションを実装