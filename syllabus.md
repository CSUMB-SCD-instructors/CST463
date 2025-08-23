# CST463 (Advanced Machine Learning) Syllabus

## Summary

In this course you will learn to use advanced machine learning methods, including dimensionality reduction, ensemble methods, neural nets, and methods for working with time series data.

[//]: # (- **Grading:**)

[//]: # (There are three assignment groups that you must achieve a minimum grade of 40% in to pass the course.)

[//]: # (They are [Programming Assignments]&#40;#programming-assignments&#41; &#40;35%&#41;, [Exams]&#40;#exams&#41; &#40;35%&#41;, and [Participation]&#40;#participation&#41; &#40;30%&#41;.)

[//]: # (Final grades are calculated by rounding to the nearest whole number and converted to letter grades using the standard range.)

[//]: # (*Details can be found in [grading]&#40;#grading&#41;.*)

[//]: # ()
[//]: # (- **Late work:**)

[//]: # (Late submissions are accepted only for Programming assignments and Labs, and have a 10% deduction per calendar day late.)

[//]: # (*Details can be found in [late policy]&#40;#late-policy&#41;.*)

[//]: # ()
[//]: # (- **Getting help:**)

[//]: # (There are a ton of office hours spread throughout the week so make use of them if you have any questions!)

[//]: # (In general, you can use the [class slack channel]&#40;#&#41; to ask questions &#40;but not post code!&#41;, or send me questions &#40;code okay!&#41; via slack or my email.)

[//]: # (You are **not** allowed to use LLMs or classmates to write code for you, but you can ask them clarifying questions about material.)

[//]: # (*Details can be found in [personnel]&#40;#personnel&#41;.*)

[//]: # ()
[//]: # (**Any other questions?**)

[//]: # (I tried to make this document exhaustive &#40;and it just turned out exhausting...&#41;, but if there's something you can't find an answer to don't hesitate to reach out via slack or email!)

## Course Information

### Course Description

In this course you will learn to use advanced machine learning methods, including dimensionality reduction, ensemble methods, neural nets, and methods for working with time series data.

### Course Objectives

At the end of this class, you should be able to do the following:
- Apply deep learning to machine-learning problems of moderate complexity, using Google’s TensorFlow library and Python.
- Explain and implement machine learning algorithms using loss functions and optimization, including the use of backpropagation in neural nets.
- Explain the principles behind dimensionality reduction and ensemble methods (including random forests and boosting).
- Apply machine learning to time series data sets.

### Topics Covered

The major topics covered in class are:
1. Working with Data & Math
2. Gradient Descent
3. Neural Network Basics
4. TensorFlow
5. Vision Models
6. Time Series
7. Text Models

Details can be found in the [course calendar](calendar.md)


### Personnel

- Dr. Sam Ogden (instructor)
  - E-mail: [`sogden@csumb.edu`](mailto:sogden@csumb.edu)
  - Web: [**`https://csumb.edu/scd/sogden/`**](https://csumb.edu/scd/sogden/)
  - Office: BIT205
  - Office Hours: 2pm-3pm Mondays & 11am-12noon Thursdays, or [by appointment](https://calendly.com/sogden-csumb/15min)

### Materials

- Course textbook: (both available on O'Reilly Media for free)
  - Deep Learning with Python, second edition.  Francois Chollet, Manning, 2021. (primary text)
  - Hands-on Machine Learning with Scikit-Learn, Keras, and Tensorflow, third edition.  Aurélien Géron, O'Reilly Media, 2019.
- Course lecture recordings: [Recording location](https://csumb.hosted.panopto.com/Panopto/Pages/Sessions/List.aspx?folderID=9c2a56f8-c79e-431c-ace6-b1d801434552)
- Canvas Course Page: [Canvas link](https://csumb.instructure.com/courses/29740)
- Syllabus: [syllabus.md](syllabus.md)

### Prerequisites/Corequisites

- **CST383**: Introduction to Data Science 

## Grading

There are three groups of assignments, briefly described in the table below.
**Getting below a 40% in any of these three groups will result in failing the class.**

| Assignment Kind                                     | Value |
|:----------------------------------------------------|:-----:|
| [Programming assignments](#programming-assignments) |  36%  |
| [Exams](#exams)                                     |  36%  |
| Group Project                                       |  16%  |
| [Participation](#participation)                     |  12%  |

### Programming Assignments

Programming assignments will consist of three parts:
- Development of a library of functions that perform a set of tasks
- Visualization and analysis using the functions developed during the first half
- Reviews of your classmates work

The goal of these three parts is to give you experience in the three major aspects of data analysis and machine learning: 
development of new module and code,
analysis of data,
interpreting the work of others.

Assignments will generally span two weeks, with the first week focusing on the development of library functions and review of the previous week's analysis, and the second week will focus on your own analysis and visualization.

Therefore, there will be three parts to how you are assess:
1. The python code you develop will be checked against unit tests to ensure expected functionality (roughly 50%)
2. Your visualization and analysis, based on how well I and your peers understand your interpretation and the validity (roughly 30%)
3. Your analysis of your peer's visualization and analysis. (roughly 20%)


### Exams

There will be three exams in this course worth each worth roughly 12% of your grade.
Each one will cover a mixture of new and old material, with an emphasis on new material.
During the exam you will be allowed to use a calculator but no other electronics, and have a single sheet of notes.

These exams are spaced roughly 5 weeks apart and can be seen on the course [calendar](calendar.md).

Note: there is no final exam for this class, but we may have a project presentations period during this block.

### Participation

In-class participation has three components:

1. (5%) Learning Logs: reflect on the material you learned each week and use as notes for studying
2. (5%) Attendance: Show up to class on time every day
3. (2%) Weekly Surveys: to help me understand how to improve class

### Extra Credit Potential

#### Introductory Meeting (2%)

During the first month of class, stop by my office and chat for a bit to get to know each other and you’ll get a small number of bonus points.
You can either stop by my office during my regularly scheduled hours or schedule an appointment.
See details in [the personnel section](#personnel) to find out how to schedule a meeting.

#### Bug Hunting

We all make mistakes – my github commit history is proof of that.
Sometimes you catch big ones before I do, and if you let me know I might throw some extra credit your way.
This is generally reserved for big things (e.g. I left out a bunch of files in the repo or some tools out of the docker image), but if you find yourself confused by something please reach out, maybe you found a novel bug!
[^1]

### Grade Assignment

Grades as assigned based on the standard ranges, which are outlined below.

| Grade Range     | Letter Grade |
|:----------------|:-------------|
| $[97,\infty)$   | A+           |
| $[93,97)$       | A            |
| $[90,93)$       | A-           |
| $[87,90)$       | B+           |
| $[83,87)$       | B            |
| $[80,83)$       | B-           |
| $[77,80)$       | C+           |
| $[73,77)$       | C            |
| $[70,73)$       | C-           |
| $[60,70)$       | D            |
| $[0,60)$        | F            |

## Course Policies

### Attendance

Students are expected to arrive to class on time and to attend every class.
Class attendance is one of the strongest factors for student performance. 
Attendance will be tracked either manually or with an attendance quiz.
While no retakes of a quiz or attendance will be allowed, the lowest 2 will be dropped.

### Communication

#### Contacting me

You can reach me via email (sogden@csumb.edu, or via slack by either DM or tagging me in a message to the channel.
In general, I will not see or respond to canvas messages/comments so please use slack or email to contact me.

#### Communicating online

I will use email, Canvas, and Slack for online class communication.

1.  **email** is to be used when you need to officially communicate something to me.
  - e.g. “Can you double check this assignment grade for me?”
2.  **Slack** will be used to answer content questions and give quick updates.
  - e.g. “I think I found a bug in this quiz, could you check it out?”
  - We have a slack channel named [\#cst463-fall2025](https://cs-you-monterey.slack.com/archives/C09BPTUFLAH) on the [CS-U-Monterey](https://cs-you-monterey.slack.com/archives/C04LQ29KWV6) workspace that you should join.
3.  **Canvas** will have information on assignments and due dates
  - In general I don’t see canvas messages, and the likelihood of me seeing a comment on an assignment is extremely low. If you have questions for comments for me please reach out via either slack or email.

### Late Policy

No late work will be accepted and no extensions will be granted.

[//]: # (These two exceptions are programming assignments and lab assignments, both of which can be submitted with a late penalty at any point before the Sunday prior to final exam week.)

[//]: # (This late penalty will be a 10% reduction in maximum points per day, with a maximum reduction of 60%[^3].)

**If you do fall behind in class, please set up a time to meet with me to discuss getting you back on track – my goal is to have you succeed in class and I want to find a way to make that happen.**

## Academic Honor Code

In addition to the department- and school-wide policies on academic honesty outlined elsewhere on canvas, I wanted to emphasize a few key points about expectations in class.

### Motivation

This class aims to help you learn how to approach systems problems in computer science, consider alternatives, and develop effective solutions.
In your career, it will be important to not just be able to write code, but also to think about how this code fits into a larger ecosystem and the trade-offs being made.
These are skills that are essential, and are not developed in isolation, but need to be core skills you can demonstrate during interviews and jobs.
Therefore, you should learn to search out resources, understand when to ask for more, but also be capable of applying these approaches on your own.

### Summary of policy

**You may talk to other students or LLMs (e.g. ChatGPT) for:**

- Clarification on topics covered in class (e.g. paging, API usage, etc.)
- Clarification of what *provided* code is doing
- Generating more examples
- Understanding compiler errors

**You may *not* talk to other students or LLMs for:**

- Code and checkpoints answers – i.e. do not copy-and-paste code or answers from other students or LLMs
- Exam questions and answers

### Getting Help

If you have problems with the homework, please ask the instructor or TA for help.
On coding assignments, you may also get help from fellow students, but you must follow the [Stanford Honor Code](https://communitystandards.stanford.edu/policies-guidance/honor-code)

1.  You *must not* look at solutions or program code that are not your own.
2.  You *must not* share your solution code with others, and must maintain the privacy of your solution.
3.  You *must* indicate on your submission any assistance you received.

To enforce this, I will typically read your code and use a code similarity checker.
If I have concerns, I will let you know, either through a comment on canvas or an email.

### The use of LLMs

At the highest level, the use of LLMs in this class is ***prohibited** for programming assignments* and ***encouraged** for improving your own understanding of material*.
Large Language Models (LLMs), such as ChatGPT and GPT4.0, are powerful language generation models.
They can be used to produce code and summarize text, as well as answer clarifying questions.
They can be a very effective tool in a programmer’s toolbox if used appropriately.
In this class, you should use them as a resource similar to [stackoverflow](https://stackoverflow.com/) – a good resource that should be critically considered since the code might be wrong, and not as effective as coming to talk to a TA or instructor.

**You may use** LLMs for asking clarifying questions and generating examples.
I will demonstrate a number of these such questions in class but “how do I use XX command?” and “how does a free list in memory management work?” are examples.
Additionally, asking for example problems (e.g. “can I have five examples of turnaround time calculation with FIFO and RR scheduling?”) is an excellent use[^4].
Further, if you are confused about what a homework problem is asking or how it relates to operating systems at a larger scale you can ask them[^5].

**You may *not*** use LLMs for producing answers on programming assignments, quizzes or exams.
Asking for an LLM’s solution to a coding assignment is conceptually the same as asking another student to write you code for you, and will be treated as such.
While you may ask for individual parts to help understand the C language better (e.g. “how do I write a do-while loop in C”), using it for more than ~2 lines of code is not allowed.
As a rule of thumb, a prompt of “write me a function to do...” indicates that you are headed in a problematic direction.

You are encouraged to use LLMs like you would a TA or an instructor.
You *may not* use LLMs for producing code for programming assignments, but *may* use it to learn how to better use the language and to clarify topics.
They can clarify complex topics we learned in class, give you alternative explanations and can be incredibly helpful in understanding errors and problems.
However, it is important to be able to identify *good* examples and *bad* examples of LLM output.

### COVID-19 Classroom Policy

Please familiarize yourself with the CSUMB COVID-19 policies, which can be found [here](https://csumb.edu/health/coronavirus-information/).
In short, campus, and class, is a mask-friendly environment.
If you have been exposed to COVID-19 or are COVID-19 positive please stay home from class and contact me to arrange to attend virtually.
This also generally applies when you are ill due to other causes – instead of coming to class while sick, please reach out, and we can make a plan.


[^1]: As a heads up, don’t edit your unit tests until after you’ve checked with me! If it’s going to lead to changes in the test I’ll make sure there’s an announcement about it!

[^2]: “\[C\]lass attendance \[is\] a better predictor of college grades than any other known predictor of academic performance” from <https://doi.org/10.3102/0034654310362998>

[^4]: Although LLMs are notoriously bad at math so be very wary of their calculations.

[^5]: Although I recommend you talk to the TAs or the instructor since we likely have a better idea of what the assignment is asking.

