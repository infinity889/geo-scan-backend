$ErrorActionPreference = "Stop"

$binDir = "C:\Users\lenovo\OneDrive\Desktop\geo-scan\geo-scan-backend\bin"
if (-not (Test-Path $binDir)) {
    New-Item -ItemType Directory -Path $binDir | Out-Null
}

Write-Host "Downloading Poppler..."
$popplerZip = "$binDir\poppler.zip"
Invoke-WebRequest -Uri "https://github.com/oschwartz10612/poppler-windows/releases/download/v24.02.0-0/Release-24.02.0-0.zip" -OutFile $popplerZip
Write-Host "Extracting Poppler..."
Expand-Archive -Path $popplerZip -DestinationPath $binDir -Force
Rename-Item -Path "$binDir\poppler-24.02.0" -NewName "poppler" -ErrorAction SilentlyContinue
Remove-Item -Path $popplerZip -Force

Write-Host "Downloading Tesseract..."
$tesseractExe = "$binDir\tesseract_setup.exe"
Invoke-WebRequest -Uri "https://github.com/UB-Mannheim/tesseract/releases/download/v5.4.0.20240606/tesseract-ocr-w64-setup-5.4.0.20240606.exe" -OutFile $tesseractExe

Write-Host "Installing Tesseract to $binDir\tesseract..."
$tesseractDir = "$binDir\tesseract"
Start-Process -FilePath $tesseractExe -ArgumentList "/S", "/D=$tesseractDir" -Wait
Remove-Item -Path $tesseractExe -Force

Write-Host "Downloading rus.traineddata for Tesseract..."
$tessdataDir = "$tesseractDir\tessdata"
if (-not (Test-Path $tessdataDir)) {
    New-Item -ItemType Directory -Path $tessdataDir | Out-Null
}
Invoke-WebRequest -Uri "https://github.com/tesseract-ocr/tessdata/raw/main/rus.traineddata" -OutFile "$tessdataDir\rus.traineddata"

Write-Host "Done!"
