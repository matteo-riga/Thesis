import Spell

def process_spell_log_key(log_list):
    s = Spell.Spell()
    s.load_templates_from_file()
    s.parse_logs(log_list)
    s.save_template_to_file()


# manager = TemplateManager('templates.txt')
# manager.write_list_to_file()
# manager.load_templates_from_file()
# manager.save_template_to_file()