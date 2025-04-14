import qrcode

urls = {
    "github": "https://github.com/KylianSchmidt/aido",
    "arxiv": "https://arxiv.org/pdf/2502.02152",
}

for name, url in urls.items():
    img = qrcode.make(url)
    img.save(f"qr_{name}.png")
