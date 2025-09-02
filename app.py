from flask import Flask, request, jsonify, send_from_directory
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import re

app = Flask(__name__)

# استخدام EfficientNet بدلاً من MobileNet للحصول على دقة أفضل
model = EfficientNetB0(weights="imagenet")
print("🤖 تم تحميل النموذج بنجاح!")

# قاموس الترجمة للأصناف الشائعة
ARABIC_TRANSLATIONS = {
    # الحيوانات
    'egyptian_cat': 'قطة مصرية',
    'tabby': 'قطة مخططة',
    'tiger_cat': 'قطة نمرية',
    'persian_cat': 'قطة فارسية',
    'siamese_cat': 'قطة سيامية',
    'cat': 'قطة',
    'kitten': 'قطة صغيرة',
    
    'golden_retriever': 'كلب جولدن ريتريفر',
    'german_shepherd': 'كلب الراعي الألماني',
    'beagle': 'كلب بيجل',
    'bulldog': 'كلب بولدوج',
    'poodle': 'كلب بودل',
    'chihuahua': 'كلب تشيهواهوا',
    'labrador_retriever': 'كلب لابرادور',
    'dog': 'كلب',
    'puppy': 'جرو',
    
    'horse': 'حصان',
    'cow': 'بقرة',
    'sheep': 'خروف',
    'pig': 'خنزير',
    'goat': 'ماعز',
    'chicken': 'دجاجة',
    'duck': 'بطة',
    'goose': 'إوزة',
    'turkey': 'ديك رومي',
    
    'elephant': 'فيل',
    'lion': 'أسد',
    'tiger': 'نمر',
    'bear': 'دب',
    'zebra': 'حمار وحشي',
    'giraffe': 'زرافة',
    'monkey': 'قرد',
    'snake': 'ثعبان',
    'bird': 'طائر',
    'eagle': 'نسر',
    'owl': 'بومة',
    'parrot': 'ببغاء',
    'fish': 'سمكة',
    'shark': 'قرش',
    'whale': 'حوت',
    'dolphin': 'دولفين',
    'turtle': 'سلحفاة',
    'frog': 'ضفدع',
    'spider': 'عنكبوت',
    'butterfly': 'فراشة',
    'bee': 'نحلة',
    'ant': 'نملة',
    
    # الأشخاص
    'person': 'شخص',
    'man': 'رجل',
    'woman': 'امرأة',
    'child': 'طفل',
    'baby': 'رضيع',
    'boy': 'ولد',
    'girl': 'بنت',
    'face': 'وجه',
    
    # المباني والأماكن
    'house': 'بيت',
    'building': 'مبنى',
    'church': 'كنيسة',
    'mosque': 'مسجد',
    'castle': 'قلعة',
    'tower': 'برج',
    'bridge': 'جسر',
    'road': 'طريق',
    'street': 'شارع',
    'park': 'حديقة',
    'garden': 'بستان',
    'beach': 'شاطئ',
    'mountain': 'جبل',
    'forest': 'غابة',
    'desert': 'صحراء',
    'lake': 'بحيرة',
    'river': 'نهر',
    'sea': 'بحر',
    'ocean': 'محيط',
    'sky': 'سماء',
    'cloud': 'سحابة',
    'sun': 'شمس',
    'moon': 'قمر',
    'star': 'نجمة',
    
    # المركبات
    'car': 'سيارة',
    'truck': 'شاحنة',
    'bus': 'حافلة',
    'motorcycle': 'دراجة نارية',
    'bicycle': 'دراجة هوائية',
    'airplane': 'طائرة',
    'helicopter': 'هليكوبتر',
    'boat': 'قارب',
    'ship': 'سفينة',
    'train': 'قطار',
    
    # الطعام
    'apple': 'تفاحة',
    'banana': 'موزة',
    'orange': 'برتقالة',
    'strawberry': 'فراولة',
    'grape': 'عنب',
    'pizza': 'بيتزا',
    'burger': 'برجر',
    'bread': 'خبز',
    'cake': 'كيكة',
    'coffee': 'قهوة',
    'tea': 'شاي',
    'water': 'ماء',
    'milk': 'حليب',
    'cheese': 'جبنة',
    'meat': 'لحم',
    'chicken_meat': 'لحم دجاج',
    'fish_food': 'سمك (طعام)',
    'rice': 'أرز',
    'pasta': 'مكرونة',
    'salad': 'سلطة',
    'soup': 'شوربة',
    'ice_cream': 'آيس كريم',
    'chocolate': 'شوكولاتة',
    'candy': 'حلوى',
    
    # الأدوات والأشياء
    'phone': 'هاتف',
    'computer': 'كمبيوتر',
    'laptop': 'لابتوب',
    'television': 'تلفزيون',
    'camera': 'كاميرا',
    'book': 'كتاب',
    'pen': 'قلم',
    'pencil': 'قلم رصاص',
    'paper': 'ورق',
    'bag': 'حقيبة',
    'chair': 'كرسي',
    'table': 'طاولة',
    'bed': 'سرير',
    'door': 'باب',
    'window': 'نافذة',
    'mirror': 'مرآة',
    'clock': 'ساعة',
    'watch': 'ساعة يد',
    'key': 'مفتاح',
    'lamp': 'مصباح',
    'flower': 'زهرة',
    'tree': 'شجرة',
    'grass': 'عشب',
    'leaf': 'ورقة شجر',
    'stone': 'حجر',
    'rock': 'صخرة',
    'sand': 'رمل',
    'snow': 'ثلج',
    'ice': 'جليد',
    'fire': 'نار',
    'smoke': 'دخان',
    
    # الرياضة
    'football': 'كرة قدم',
    'basketball': 'كرة سلة',
    'tennis_ball': 'كرة تنس',
    'golf_ball': 'كرة جولف',
    'baseball': 'بيسبول',
    'volleyball': 'كرة طائرة',
    
    # الملابس
    'shirt': 'قميص',
    'pants': 'بنطال',
    'dress': 'فستان',
    'shoes': 'أحذية',
    'hat': 'قبعة',
    'jacket': 'جاكيت',
    'tie': 'ربطة عنق',
    'socks': 'جوارب',
    'gloves': 'قفازات',
    'belt': 'حزام',
}

def clean_label(label):
    """تنظيف وتحسين التسمية"""
    # إزالة الأرقام والرموز الخاصة
    cleaned = re.sub(r'[0-9_-]', ' ', label.lower())
    # إزالة المسافات الزائدة
    cleaned = ' '.join(cleaned.split())
    return cleaned.strip()

def get_arabic_translation(english_label):
    """الحصول على الترجمة العربية"""
    # تنظيف التسمية
    cleaned_label = clean_label(english_label)
    
    # البحث المباشر
    if cleaned_label in ARABIC_TRANSLATIONS:
        return ARABIC_TRANSLATIONS[cleaned_label]
    
    # البحث بالكلمات المفتاحية
    for key, value in ARABIC_TRANSLATIONS.items():
        if key in cleaned_label or cleaned_label in key:
            return value
    
    # البحث بالكلمات الفردية
    words = cleaned_label.split()
    for word in words:
        if word in ARABIC_TRANSLATIONS:
            return ARABIC_TRANSLATIONS[word]
    
    # إذا لم يتم العثور على ترجمة، إرجاع النص الأصلي محسن
    return cleaned_label.title()

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "لم يتم رفع أي ملف"})
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "لم يتم اختيار أي ملف"})

    # التحقق من نوع الملف
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    if file_extension not in allowed_extensions:
        return jsonify({"error": "نوع الملف غير مدعوم. يرجى استخدام صور بصيغة JPG, PNG, GIF"})

    # إنشاء مجلد uploads إذا لم يكن موجوداً
    upload_folder = os.path.join(os.getcwd(), "uploads")
    os.makedirs(upload_folder, exist_ok=True)

    # حفظ الملف بأسم آمن
    import time
    safe_filename = f"{int(time.time())}_{file.filename}"
    filepath = os.path.join(upload_folder, safe_filename)
    file.save(filepath)

    print(f"📁 الصورة محفوظة في: {filepath}")

    try:
        # معالجة الصورة للنموذج
        img = image.load_img(filepath, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # توقع التصنيف - الحصول على أفضل 3 نتائج
        preds = model.predict(x, verbose=0)
        decoded_predictions = decode_predictions(preds, top=3)[0]
        
        # اختيار أفضل نتيجة
        best_prediction = decoded_predictions[0]
        class_id, english_label, confidence = best_prediction
        
        # الحصول على الترجمة العربية
        arabic_label = get_arabic_translation(english_label)
        
        # تنظيف التسمية الإنجليزية
        clean_english_label = clean_label(english_label).title()
        
        # إنشاء قائمة بجميع التوقعات مع الترجمات
        all_predictions = []
        for pred in decoded_predictions:
            pred_english = clean_label(pred[1]).title()
            pred_arabic = get_arabic_translation(pred[1])
            all_predictions.append({
                "english": pred_english,
                "arabic": pred_arabic,
                "confidence": float(pred[2])
            })
        
        # حذف الملف بعد المعالجة
        try:
            os.remove(filepath)
        except:
            pass
        
        return jsonify({
            "english_label": clean_english_label,
            "arabic_label": arabic_label,
            "confidence": float(confidence),
            "all_predictions": all_predictions,
            "success": True
        })
        
    except Exception as e:
        # حذف الملف في حالة الخطأ
        try:
            os.remove(filepath)
        except:
            pass
            
        print(f"❌ خطأ في معالجة الصورة: {str(e)}")
        return jsonify({
            "error": f"خطأ في معالجة الصورة: {str(e)}",
            "success": False
        })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("🚀 بدء تشغيل خادم تصنيف الصور...")
    print("📱 الرابط: http://0.0.0.0:{}".format(port))
    app.run(debug=False, host='0.0.0.0', port=port)