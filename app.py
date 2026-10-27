import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# SAMPLE answers: replace with real info about your university
FAQS = [
    ("Hello hi hey",
     "Hi! I'm the ABES Engineering College assistant. Ask me about courses, admissions, hostels, library, placements and campus life."),
    ("Thank you thanks",
     "You're welcome! Let me know if you have any other questions."),
    ("Tell me about ABES Engineering College",
     "ABES Engineering College (ABESEC) is a private engineering college in Ghaziabad, Uttar Pradesh, established in 2000. It is approved by AICTE and affiliated with AKTU, with a campus of about 17 acres."),
    ("Where is the college located? address",
     "The college is at 19th KM Stone, NH-09, Ghaziabad, Uttar Pradesh."),
    ("What courses does ABES offer? branches programs",
     "ABES offers B.Tech in branches such as CSE, IT, AI, AI & ML, Data Science, Electronics & Communication, Electrical, Mechanical and Civil, along with M.Tech, MBA and MCA programs."),
    ("How can I apply for admission? application form",
     "You can apply online through the Admission Enquiry form on the official website (abes.ac.in)."),
    ("How is B.Tech admission done? UPTAC JEE Main counselling",
     "B.Tech admission is mainly through UPTAC/AKTU counselling based on JEE Main scores. Direct admission is also possible through the online enquiry form, subject to eligibility."),
    ("MCA admission entrance exam",
     "MCA admission requires a valid UPCET or CUET PG score."),
    ("What is the fee structure? tuition fees cost",
     "Fees depend on the course and category. Please check the fee structure on the official website or contact the admission office."),
    ("Does the college provide scholarships?",
     "Students can apply for various scholarships, including government scholarships. Please check the official website or the student welfare office for details."),
    ("Hostel facilities accommodation rooms",
     "ABES has 6 boys' and 2 girls' hostels with AC and non-AC rooms in single, double and triple sharing. Each hostel has a gym, laundry, mess and canteen, plus sports courts."),
    ("Library facilities books e-journals",
     "The library covers 1384 sq.m with a 400 sq.m reading area, over 112,000 books, 1224 e-journals, 8074 e-books and seating for 250. It is fully automated and offers remote access to e-resources."),
    ("Canteen food mess dining",
     "There are seven canteens, five on campus and two in the hostels. The seven messes and kitchens can serve around 2000 students."),
    ("Sports facilities gym swimming pool stadium",
     "The campus has a swimming pool, a floodlit cricket stadium, and courts for badminton, lawn tennis, basketball, table tennis, soccer and volleyball. Chess, athletics and snooker are also available."),
    ("Clubs and societies activities",
     "ABES has clubs for yoga, meditation, literature, cultural activities, creativity, sports, debate, publication and magazine, environment, social activity, NSS, photography, mental well-being and tourism."),
    ("College fests events Genero cultural fest",
     "Genero is the annual cultural fest, with cultural, literary, e-gaming and photography events. The college also holds a tech fest and other events such as Utsaah and TEDx."),
    ("Medical facility doctor ambulance health",
     "The campus has a medical room with a doctor and ambulance service, along with spaces for yoga and meditation."),
    ("Counselling mental health support",
     "ABES offers confidential counselling through its 'Your Dost' sessions for personal and academic support."),
    ("Placements recruiters campus drives",
     "ABES has a dedicated placement cell. The process generally includes pre-placement talks, written tests, group discussions and interviews. Please check the official website for the latest placement figures."),
    ("NCC National Cadet Corps",
     "The ABES NCC unit was established on 25 November 2024 under the 37th UP Battalion, Ghaziabad."),
    ("When are the semester exams held? exam schedule admit card",
     "Semester exams follow the AKTU schedule. Please check official college notices and the student portal for dates and admit cards."),
    ("How can I contact the college? email phone",
     "You can reach the college at webadmin@abes.ac.in or through the official website, abes.ac.in. The campus is at 19th KM Stone, NH-09, Ghaziabad."),
]
THRESHOLD = 0.3  # minimum similarity to trust a match

questions = [q for q, _ in FAQS]
vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True)
faq_matrix = vectorizer.fit_transform(questions)


def get_answer(user_text):
    query_vec = vectorizer.transform([user_text])
    scores = cosine_similarity(query_vec, faq_matrix)[0]
    best = scores.argmax()
    if scores[best] < THRESHOLD:
        return ("Sorry, I don't have information on that yet. "
                "Please contact the university office for help.")
    return FAQS[best][1]


st.set_page_config(page_title="University Chatbot", page_icon="🎓")
st.title("🎓 ABES Engineering College Chatbot")
st.caption("Ask about admissions, fees, hostels, exams, placements and more.")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! How can I help you today?"}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Type your question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    reply = get_answer(prompt)
    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.write(reply)