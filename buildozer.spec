[app]
title = DocumentScanner
package.name = docscan
package.domain = org.example
version = 0.1

source.dir = .
source.include_exts = py,png,jpg

requirements = python3==3.10,kivy,opencv,numpy,pillow,android

android.ndk = 25b
android.api = 31
android.ndk_api = 21
android.archs = arm64-v8a, armeabi-v7a

android.enable_androidx = True
android.permissions = CAMERA, READ_EXTERNAL_STORAGE, WRITE_EXTERNAL_STORAGE
android.numeric_version = 1
p4a.python_version = 3.10

android.sdk_path = /usr/local/lib/android/sdk
android.p4a_branch = release-2024.01.21
