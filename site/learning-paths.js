(function () {
  var root = document.documentElement;
  var media = window.matchMedia ? window.matchMedia('(prefers-color-scheme: dark)') : null;

  function savedTheme() {
    try {
      var value = localStorage.getItem('theme');
      return value === 'light' || value === 'dark' ? value : null;
    } catch (error) {
      return null;
    }
  }

  function systemTheme() {
    return media && media.matches ? 'dark' : 'light';
  }

  function updateThemeIcon() {
    var icon = document.getElementById('themeIcon');
    var button = document.getElementById('themeToggle');
    var theme = root.getAttribute('data-theme');
    if (icon) icon.textContent = theme === 'light' ? 'N' : 'D';
    if (button) button.setAttribute('aria-label', theme === 'light' ? 'Switch to dark theme' : 'Switch to light theme');
  }

  function applyTheme(theme) {
    root.setAttribute('data-theme', theme);
    updateThemeIcon();
  }

  function storeTheme(theme) {
    try {
      localStorage.setItem('theme', theme);
    } catch (error) {}
  }

  function careerGuideFromHash(hash) {
    if (!hash || hash.indexOf('#career-route-') !== 0) return null;
    var id = '';
    try {
      id = decodeURIComponent(hash.slice(1));
    } catch (error) {
      return null;
    }
    var guide = document.getElementById(id);
    return guide && guide.matches('details.career-guide') ? guide : null;
  }

  function syncCareerChoice(guide) {
    document.querySelectorAll('[data-career-choice]').forEach(function (link) {
      if (guide && link.getAttribute('href') === '#' + guide.id) link.setAttribute('aria-current', 'location');
      else link.removeAttribute('aria-current');
    });
  }

  function careerScrollBehavior() {
    return window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth';
  }

  function closeOtherCareerGuides(activeGuide) {
    document.querySelectorAll('details.career-guide').forEach(function (guide) {
      if (guide !== activeGuide) guide.open = false;
    });
  }

  function revealCareerGuide(hash, moveFocus) {
    var guide = careerGuideFromHash(hash);
    if (!guide) return false;
    closeOtherCareerGuides(guide);
    guide.open = true;
    syncCareerChoice(guide);
    window.requestAnimationFrame(function () {
      var summary = guide.querySelector('summary');
      if (moveFocus && summary) {
        try {
          summary.focus({ preventScroll: true });
        } catch (error) {
          summary.focus();
        }
      }
      guide.scrollIntoView({
        behavior: careerScrollBehavior(),
        block: 'start'
      });
    });
    return true;
  }

  applyTheme(savedTheme() || systemTheme());

  function getLang() {
    if (typeof window.AIFS_currentLang === 'function') return window.AIFS_currentLang();
    try {
      var saved = localStorage.getItem('lang');
      if (saved) return saved;
    } catch (_) {}
    return document.documentElement.lang || 'en';
  }

  function getI18n() {
    var lang = getLang();
    var bundle = (window.AIFS_UI_I18N && window.AIFS_UI_I18N[lang]) || (window.AIFS_UI_I18N && window.AIFS_UI_I18N.en) || {};
    return bundle.learningPaths || {};
  }

  function setListItems(list, values) {
    if (!list || !values || !values.length) return;
    var items = list.querySelectorAll('li');
    for (var i = 0; i < values.length && i < items.length; i++) {
      items[i].textContent = values[i];
    }
  }

  function setLabeledParagraph(paragraph, label, body) {
    if (!paragraph || !label || !body) return;
    paragraph.textContent = '';
    var strong = document.createElement('strong');
    strong.textContent = label;
    paragraph.appendChild(strong);
    paragraph.appendChild(document.createTextNode(' ' + body));
  }

  function applyCareerGuide(guide, i18n, career) {
    if (!career) return;

    var heading = guide.querySelector('.career-guide-heading');
    if (heading) {
      var wfSpan = heading.firstElementChild;
      if (wfSpan && wfSpan.tagName === 'SPAN' && i18n.workFamily) wfSpan.textContent = i18n.workFamily;
      var title = heading.querySelector('strong');
      if (title && career.title) title.textContent = career.title;
      var aliases = heading.querySelector('small');
      if (aliases && career.aliases) aliases.textContent = career.aliases;
    }

    var disclosureSpans = guide.querySelectorAll('.career-guide-disclosure span');
    if (disclosureSpans.length >= 2) {
      if (i18n.openGuide) disclosureSpans[0].textContent = i18n.openGuide;
      if (i18n.closeGuide) disclosureSpans[1].textContent = i18n.closeGuide;
    }

    var mission = guide.querySelector('.career-guide-mission');
    if (mission && career.mission) mission.textContent = career.mission;

    var dts = guide.querySelectorAll('dl dt');
    var dds = guide.querySelectorAll('dl dd');
    if (dts.length >= 2) {
      if (i18n.guidedRouteLabel) dts[0].textContent = i18n.guidedRouteLabel;
      if (i18n.baselineLabel) dts[1].textContent = i18n.baselineLabel;
    }
    if (dds.length >= 2) {
      if (career.route) dds[0].textContent = career.route;
      if (career.baseline) dds[1].textContent = career.baseline;
    }

    var sections = guide.querySelectorAll('.career-guide-grid > section');
    if (sections.length >= 4) {
      var ownTitle = sections[0].querySelector('h4');
      if (ownTitle && i18n.whatYouOwn) ownTitle.textContent = i18n.whatYouOwn;
      setListItems(sections[0].querySelector('ul'), career.own);

      var fitTitle = sections[1].querySelector('h4');
      if (fitTitle && i18n.fitAndBoundary) fitTitle.textContent = i18n.fitAndBoundary;
      var fitParagraphs = sections[1].querySelectorAll('p');
      if (fitParagraphs.length >= 2) {
        setLabeledParagraph(fitParagraphs[0], i18n.goodFitIf, career.fit);
        setLabeledParagraph(fitParagraphs[1], i18n.boundaryLabel, career.boundary);
      }

      var proofTitle = sections[2].querySelector('h4');
      if (proofTitle && i18n.portfolioProof) proofTitle.textContent = i18n.portfolioProof;
      var proofP = sections[2].querySelector('p');
      if (proofP && career.portfolio) proofP.textContent = career.portfolio;
      setListItems(sections[2].querySelector('ul'), career.evidence);

      var coverageTitle = sections[3].querySelector('h4');
      if (coverageTitle && i18n.coverageAndGaps) coverageTitle.textContent = i18n.coverageAndGaps;
      var coverageParagraphs = sections[3].querySelectorAll('p');
      if (coverageParagraphs.length >= 2) {
        if (career.coverage) coverageParagraphs[0].textContent = career.coverage;
        setLabeledParagraph(coverageParagraphs[1], i18n.stillEarnedLabel, career.stillEarned);
      }
    }

    var footerP = guide.querySelector('.career-guide-footer p');
    if (footerP && career.footer) footerP.textContent = career.footer;
    var cta = guide.querySelector('.career-guide-cta');
    if (cta && i18n.studySpecialist) cta.textContent = i18n.studySpecialist;
  }

  function applyDomainSection(section, i18n, domain) {
    if (!domain) return;

    var number = section.querySelector('.learning-paths-domain-number');
    if (number && domain.number) number.textContent = domain.number;
    var title = section.querySelector('.skills-domain-header h2');
    if (title && domain.title) title.textContent = domain.title;
    var copy = section.querySelector('.skills-domain-header > p');
    if (copy && domain.copy) copy.textContent = domain.copy;

    var rootStrong = section.querySelector('.skills-domain-root strong');
    if (rootStrong && domain.root) rootStrong.textContent = domain.root;
    var rootMeta = section.querySelector('.learning-paths-path-meta');
    if (rootMeta && domain.meta) rootMeta.textContent = domain.meta;

    var nodes = section.querySelectorAll('.skills-node');
    var nodeData = domain.nodes || [];
    for (var i = 0; i < nodes.length && i < nodeData.length; i++) {
      var index = nodes[i].querySelector('.learning-paths-node-index');
      var nodeTitle = nodes[i].querySelector('strong');
      var nodeDesc = nodes[i].querySelector('p');
      if (index && nodeData[i].index) index.textContent = nodeData[i].index;
      if (nodeTitle && nodeData[i].title) nodeTitle.textContent = nodeData[i].title;
      if (nodeDesc && nodeData[i].desc) nodeDesc.textContent = nodeData[i].desc;
    }

    var footerSpan = section.querySelector('.skills-domain-footer > span');
    if (footerSpan) {
      if (domain.footerNote) footerSpan.textContent = domain.footerNote;
      else if (i18n.followConnected) footerSpan.textContent = i18n.followConnected;
    }
  }

  function applyLearningPathsTranslations() {
    var i18n = getI18n();
    if (!i18n.heroTitle) return;

    if (i18n.pageTitle) document.title = i18n.pageTitle;

    var skipLink = document.querySelector('.skip-link');
    if (skipLink && i18n.skipLink) skipLink.textContent = i18n.skipLink;

    var navLinks = document.querySelectorAll('.header-nav > a');
    if (navLinks.length >= 5) {
      if (i18n.navContents) navLinks[0].textContent = i18n.navContents;
      if (i18n.navCatalog) navLinks[1].textContent = i18n.navCatalog;
      if (i18n.navRoadmap) navLinks[2].textContent = i18n.navRoadmap;
      if (i18n.navGlossary) navLinks[3].textContent = i18n.navGlossary;
      if (i18n.navAbout) navLinks[4].textContent = i18n.navAbout;
    }

    var eyebrow = document.querySelector('.learning-paths-eyebrow');
    if (eyebrow && i18n.heroEyebrow) eyebrow.textContent = i18n.heroEyebrow;
    var heroTitle = document.getElementById('learningPathsTitle');
    if (heroTitle && i18n.heroTitle) heroTitle.textContent = i18n.heroTitle;
    var heroLede = document.querySelector('.learning-paths-lede');
    if (heroLede && i18n.heroLede) heroLede.textContent = i18n.heroLede;

    var entryNavLinks = document.querySelectorAll('.learning-paths-entry-nav a');
    if (entryNavLinks.length >= 2) {
      if (i18n.exploreByKnowledge) entryNavLinks[0].querySelector('span').textContent = i18n.exploreByKnowledge;
      if (i18n.browseFourPaths) entryNavLinks[0].querySelector('strong').textContent = i18n.browseFourPaths;
      if (i18n.exploreByOutcome) entryNavLinks[1].querySelector('span').textContent = i18n.exploreByOutcome;
      if (i18n.chooseCareerRoute) entryNavLinks[1].querySelector('strong').textContent = i18n.chooseCareerRoute;
    }

    var overviewTitle = document.getElementById('learningPathsOverviewTitle');
    if (overviewTitle && i18n.overviewTitle) overviewTitle.textContent = i18n.overviewTitle;
    var overviewP = document.querySelector('.learning-paths-overview-copy p');
    if (overviewP && i18n.overviewCopy) overviewP.textContent = i18n.overviewCopy;
    var overviewA = document.querySelector('.learning-paths-overview-copy a');
    if (overviewA && i18n.compareSixRoutes) overviewA.textContent = i18n.compareSixRoutes;

    var overviewRoot = document.querySelector('.learning-paths-overview-root');
    if (overviewRoot && i18n.rootTitle) {
      overviewRoot.childNodes[0].nodeValue = i18n.rootTitle;
      var rootSpan = overviewRoot.querySelector('span');
      if (rootSpan && i18n.rootSubtitle) rootSpan.textContent = i18n.rootSubtitle;
    }

    var domainLinks = document.querySelectorAll('.learning-paths-domain-links a');
    if (domainLinks.length >= 4) {
      if (i18n.domain1Title) domainLinks[0].querySelector('strong').textContent = i18n.domain1Title;
      if (i18n.domain1Sub) domainLinks[0].querySelector('span').textContent = i18n.domain1Sub;
      if (i18n.domain2Title) domainLinks[1].querySelector('strong').textContent = i18n.domain2Title;
      if (i18n.domain2Sub) domainLinks[1].querySelector('span').textContent = i18n.domain2Sub;
      if (i18n.domain3Title) domainLinks[2].querySelector('strong').textContent = i18n.domain3Title;
      if (i18n.domain3Sub) domainLinks[2].querySelector('span').textContent = i18n.domain3Sub;
      if (i18n.domain4Title) domainLinks[3].querySelector('strong').textContent = i18n.domain4Title;
      if (i18n.domain4Sub) domainLinks[3].querySelector('span').textContent = i18n.domain4Sub;
    }

    var careerHeader = document.querySelector('.career-routes-header');
    if (careerHeader) {
      var cEyebrow = careerHeader.querySelector('.learning-paths-domain-number');
      if (cEyebrow && i18n.careerEyebrow) cEyebrow.textContent = i18n.careerEyebrow;
      var cTitle = document.getElementById('careerRoutesTitle');
      if (cTitle && i18n.careerTitle) cTitle.textContent = i18n.careerTitle;
      var cCopy = careerHeader.querySelector('p');
      if (cCopy && i18n.careerCopy) cCopy.textContent = i18n.careerCopy;
    }

    var truthNote = document.querySelector('.career-truth-note');
    if (truthNote) {
      var truthStrong = truthNote.querySelector('strong');
      var truthP = truthNote.querySelector('p');
      if (truthStrong && i18n.readThisFirst) truthStrong.textContent = i18n.readThisFirst;
      if (truthP && i18n.truthNoteCopy) truthP.textContent = i18n.truthNoteCopy;
    }

    var foundationItems = document.querySelectorAll('.career-foundation-strip li');
    if (foundationItems.length >= 3) {
      if (i18n.step1Title) foundationItems[0].querySelector('a').textContent = i18n.step1Title;
      if (i18n.step1Desc) foundationItems[0].querySelector('p').textContent = i18n.step1Desc;
      if (i18n.step2Title) foundationItems[1].querySelector('a').textContent = i18n.step2Title;
      if (i18n.step2Desc) foundationItems[1].querySelector('p').textContent = i18n.step2Desc;
      if (i18n.step3Title) foundationItems[2].querySelector('a').textContent = i18n.step3Title;
      if (i18n.step3Desc) foundationItems[2].querySelector('p').textContent = i18n.step3Desc;
    }

    var chooser = document.querySelector('.career-chooser');
    if (chooser) {
      var chEyebrow = chooser.querySelector('.learning-paths-domain-number');
      if (chEyebrow && i18n.chooserEyebrow) chEyebrow.textContent = i18n.chooserEyebrow;
      var chTitle = document.getElementById('careerChooserTitle');
      if (chTitle && i18n.chooserTitle) chTitle.textContent = i18n.chooserTitle;
    }

    var choiceLinks = document.querySelectorAll('.career-choice-grid a');
    if (choiceLinks.length >= 6) {
      var prompts = [
        { sub: i18n.prompt1Sub, title: i18n.prompt1Title },
        { sub: i18n.prompt2Sub, title: i18n.prompt2Title },
        { sub: i18n.prompt3Sub, title: i18n.prompt3Title },
        { sub: i18n.prompt4Sub, title: i18n.prompt4Title },
        { sub: i18n.prompt5Sub, title: i18n.prompt5Title },
        { sub: i18n.prompt6Sub, title: i18n.prompt6Title }
      ];
      for (var p = 0; p < prompts.length; p++) {
        if (prompts[p].sub) choiceLinks[p].querySelector('span').textContent = prompts[p].sub;
        if (prompts[p].title) choiceLinks[p].querySelector('strong').textContent = prompts[p].title;
      }
    }

    var careers = i18n.careers || {};
    document.querySelectorAll('.career-guide').forEach(function (guide) {
      var key = guide.getAttribute('data-career-guide');
      applyCareerGuide(guide, i18n, careers[key]);
    });

    var domains = i18n.domains || {};
    document.querySelectorAll('section.skills-domain').forEach(function (section) {
      applyDomainSection(section, i18n, domains[section.id]);
    });

    document.querySelectorAll('.skills-domain-root-link').forEach(function (link) {
      if (i18n.openFullPath) link.textContent = i18n.openFullPath;
    });

    document.querySelectorAll('.skills-node-action').forEach(function (action) {
      var isRep = action.getAttribute('data-i18n-action') === 'rep' ||
        /representative|대표/.test(action.textContent || '');
      if (isRep) {
        if (i18n.openRepLesson) {
          action.textContent = i18n.openRepLesson;
          action.setAttribute('data-i18n-action', 'rep');
        }
      } else if (i18n.openLesson) {
        action.textContent = i18n.openLesson;
        action.setAttribute('data-i18n-action', 'lesson');
      }
    });

    document.querySelectorAll('.skills-domain-footer a.learning-paths-back').forEach(function (back) {
      if (i18n.backToFourDomains) back.textContent = i18n.backToFourDomains;
    });

    var careerFooterSpan = document.querySelector('.career-routes-footer span');
    if (careerFooterSpan && i18n.unsureFooter) careerFooterSpan.textContent = i18n.unsureFooter;
    var careerFooterBack = document.querySelector('.career-routes-footer a.learning-paths-back');
    if (careerFooterBack && i18n.compareFourDomains) careerFooterBack.textContent = i18n.compareFourDomains;

    var footerCopy = document.querySelector('.site-footer .footer-inner > p');
    if (footerCopy && i18n.footerCopy) footerCopy.textContent = i18n.footerCopy;
    var footerLinks = document.querySelectorAll('.site-footer .footer-links a');
    if (footerLinks.length >= 3) {
      if (i18n.footerHome) footerLinks[0].textContent = i18n.footerHome;
      if (i18n.footerRoadmap) footerLinks[1].textContent = i18n.footerRoadmap;
      if (i18n.footerCatalog) footerLinks[2].textContent = i18n.footerCatalog;
    }
  }

  // Handle dynamic language switch from lang-picker
  if (typeof window !== 'undefined') {
    window.AIFS_onLearningPathsLangChange = function (newLang) {
      document.documentElement.lang = newLang;
      applyLearningPathsTranslations();
    };
    if (typeof window.AIFS_onLangChange === 'function') {
      var prevHandler = window.AIFS_onLangChange;
      window.AIFS_onLangChange = function (l) {
        prevHandler(l);
        if (window.AIFS_onLearningPathsLangChange) window.AIFS_onLearningPathsLangChange(l);
      };
    } else {
      window.AIFS_onLangChange = function (l) {
        if (window.AIFS_onLearningPathsLangChange) window.AIFS_onLearningPathsLangChange(l);
      };
    }
  }

  document.addEventListener('DOMContentLoaded', function () {
    applyLearningPathsTranslations();
    var button = document.getElementById('themeToggle');
    if (button) {
      updateThemeIcon();
      button.addEventListener('click', function () {
        var next = root.getAttribute('data-theme') === 'light' ? 'dark' : 'light';
        applyTheme(next);
        storeTheme(next);
      });
    }

    document.querySelectorAll('[data-career-choice]').forEach(function (link) {
      link.addEventListener('click', function (event) {
        var hash = link.getAttribute('href');
        if (!careerGuideFromHash(hash)) return;
        event.preventDefault();
        if (window.location.hash === hash) revealCareerGuide(hash, true);
        else window.location.hash = hash;
      });
    });

    if (!revealCareerGuide(window.location.hash, false)) syncCareerChoice(null);
  });

  window.addEventListener('hashchange', function () {
    if (!revealCareerGuide(window.location.hash, true)) syncCareerChoice(null);
  });

  window.addEventListener('storage', function (event) {
    if (event.key === 'theme') applyTheme(savedTheme() || systemTheme());
  });

  if (media) {
    var syncSystemTheme = function () {
      if (!savedTheme()) applyTheme(systemTheme());
    };
    if (typeof media.addEventListener === 'function') media.addEventListener('change', syncSystemTheme);
    else if (typeof media.addListener === 'function') media.addListener(syncSystemTheme);
  }
})();
