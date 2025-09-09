# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Course Template Overview

This is a course template repository designed for creating new academic course repositories with automatic syllabus publishing capabilities. The template provides structured organization for programming assignments, labs, demonstrations, and automated course material management.

## Core Configuration

### Required Configuration Files
- `.course-config` - Contains course-specific variables:
  ```bash
  COURSE_CODE=CSTXXX        # Course identifier
  COURSE_NAME="Course Title" # Full course name
  CALENDAR_URL=""           # Optional external calendar URL
  ```

### Essential Files
- `syllabus.md` - Main course syllabus (triggers auto-sync to centralized repository)
- `calendar.md` - Optional local calendar file (alternative to CALENDAR_URL)
- `README.md` - Course repository documentation
- `TEMPLATE_SETUP.md` - Template maintenance and setup instructions

## Repository Structure

```
course-template/
├── .course-config          # Course configuration variables
├── syllabus.md            # Course syllabus (auto-synced)
├── calendar.md            # Optional course calendar
├── demos/                 # In-class demonstration code
├── labs/                  # Lab assignments and starter code  
├── programming-assignments/ # Major homework assignments
├── helpers/               # Grading and utility scripts
├── scripts/               # Course management scripts
│   ├── publish_to_students.sh  # Student repository publication
│   └── README.md          # Student publication workflow docs
└── .github/workflows/     # GitHub Actions automation
    └── sync-syllabus.yml  # Syllabus synchronization workflow
```

## Key Workflows

### Syllabus Publishing
- Automatic sync to centralized syllabi repository triggered by changes to:
  - `syllabus.md`
  - `calendar.md`
  - `.course-config`
- Requires `SYLLABI_SYNC_TOKEN` organization secret
- Creates Jekyll-compatible pages with proper front matter

### Student Repository Publication
The template includes a two-branch system for managing student distributions:

1. **Development**: Work on `main` branch with complete solutions
2. **Redaction**: Use `redacted_for_students` branch to remove solutions
3. **Publication**: Script publishes redacted content to student repositories

Commands for student publication workflow:
```bash
git checkout redacted_for_students
git merge main
# Manually redact solutions (replace with // todo comments)
git commit -m "Redact solutions for student distribution"
./scripts/publish_to_students.sh
```

## Template Usage

### Creating New Course from Template
1. Create repository from template
2. Update `.course-config` with course details
3. Customize `syllabus.md` content
4. Commit changes to trigger first syllabus sync

### Template Updates
Pull improvements from template to existing courses:
```bash
git remote add template https://github.com/your-org/course-template
git fetch template
git cherry-pick <improvement-commits>
```

## Programming Assignment Pedagogical Approach

Programming assignments follow a consistent two-part structure designed to reinforce both technical implementation and analytical thinking:

### Part 1: Technical Implementation
- **Core Concept Implementation**: Students implement fundamental algorithms/concepts from scratch (e.g., custom matrix multiplication, PCA pipeline)
- **Hands-on Learning**: Building implementations by hand deepens understanding of underlying mathematics and concepts
- **Scaffolding Philosophy**: Early assignments may include hints and detailed docstrings, but hints should be progressively removed in later assignments to increase difficulty
- **Testing Integration**: Implementations are validated against comprehensive test suites

### Part 2: Visualization & Analysis  
- **Applied Analysis**: Students use their implementations on real datasets via Jupyter notebooks
- **Comparative Thinking**: Compare naive/intuitive approaches against principled methods (e.g., manual feature selection vs. PCA)
- **Visual Communication**: Create clear matplotlib visualizations with proper labeling and interpretation
- **Critical Reflection**: Written analysis requiring students to articulate insights, limitations, and trade-offs
- **Peer Review Preparation**: Executive summaries and specific questions to develop communication skills

### Design Principles
- **Coherent Integration**: Both parts use the same custom implementation, reinforcing the connection between theory and practice  
- **Progressive Difficulty**: Reduce hints and increase independence across assignments throughout the semester
- **Real-World Application**: Use authentic datasets that generate meaningful insights
- **Communication Skills**: Emphasize clear explanation of technical concepts for peer audiences

This approach ensures students both understand the mathematical foundations and can apply them effectively to real problems.

## Important Notes

- No package.json, Makefile, or traditional build system - this is a content management template
- Primary automation via GitHub Actions workflows
- Course content organization follows pedagogical structure (assignments, labs, demos)
- Built for academic environments with centralized syllabus management
- Student repository publication includes safety checks and confirmation prompts